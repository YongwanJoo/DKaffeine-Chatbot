from typing import Tuple, Optional, Dict, Literal, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
import logging
import os
import re
from app.domain.ports.rag_port import RAGPort
from app.infrastructure.adapters.llm import LLMProvider
from app.infrastructure.adapters.retrievers import BaseRetriever
from app.infrastructure.config.config_loader import get_bedrock_config, get_config_float
from .rerankers import CrossEncoderReranker, BedrockReranker
from .rag_utils import extract_user_question, clean_markdown, parse_json_response, extract_filename_from_uri
from app.infrastructure.utils.token_utils import count_tokens, truncate_documents

logger = logging.getLogger(__name__)




class RAGService(RAGPort):
    """RAG 서비스 (Bedrock KB 전용)
    
    AWS Bedrock Knowledge Base를 사용한 RAG (Retrieval-Augmented Generation) 서비스입니다.
    문서 검색과 답변 생성을 통합하여 제공합니다.
    
    특징:
    - Bedrock KB 기반 벡터 검색
    - Confidence 기반 답변 필터링
    - 문서 출처 추적
    - Circuit Breaker & Retry 적용
    - 메타데이터 기반 파일명 추출 (환경 독립적)
    
    Example:
        ```python
        service = RAGService()
        has_answer, answer, sources, confidence, doc_mapping = service.search(
            query="회사 소개",
            company_id="company_001",
            top_p=0.5
        )
        ```
    
    Args:
        retriever: BaseRetriever 인스턴스 (None이면 BedrockKBRetriever 자동 생성)
    """
    
    def __init__(self, retriever: BaseRetriever | None = None, redis_client: Any | None = None):
        # Provider 통해 임베딩/LLM 주입 (Bedrock 전환 대비)
        self.provider = LLMProvider(redis_client=redis_client)
        # 기본 비즈니스용 LLM (설정은 search() 호출 시 동적으로 적용)
        self.llm = None  # 동적 생성
        # 검색기 주입 (Bedrock KB만 사용)
        if retriever:
            self.retriever = retriever
        else:
            # Bedrock KB Retriever 생성 (폴백 없음)
            from app.infrastructure.adapters.retrievers.bedrock_kb import BedrockKBRetriever
            config = get_bedrock_config()
            
            try:
                self.retriever = BedrockKBRetriever(
                    region=config.get("region"),
                    knowledge_base_id=config.get("knowledge_base_id"),
                    aws_access_key_id=config.get("access_key_id"),
                    aws_secret_access_key=config.get("secret_access_key"),
                    redis_client=redis_client,
                )
                logger.info("BedrockKBRetriever 초기화 완료")
            except Exception as e:
                logger.error(f"BedrockKBRetriever 초기화 실패: {e}")
                raise RuntimeError(
                    f"Bedrock Knowledge Base 초기화에 실패했습니다.\n"
                    f"에러: {e}\n"
                    f"`.secrets.toml` 파일 또는 환경변수를 확인하세요."
                ) from e
        
        # Perf: Bedrock Claude Haiku 4.5 사용 (LLMProvider를 통해, 응답 속도 최적화)
        self.claude_available = True
        logger.info("Bedrock Claude Haiku 4.5 available via LLMProvider")
        
        # Reranker 초기화 (지연 로딩)
        self.reranker = None
    
    def search(
        self, 
        query: str, 
        chat_history: list = None,
        top_k: int = 5,
        use_claude: bool = True,
        settings: Optional[dict] = None,
        confidence_method: Literal["reranker"] = "reranker",
        response_format: str = "plain"
    ) -> Tuple[bool, Optional[str], Optional[list], float, Dict[int, str], list, Optional[str]]:
        """RAG 검색 및 답변 생성
        
        Bedrock KB에서 관련 문서를 검색하고, LLM을 사용하여 답변을 생성합니다.
        Confidence 점수를 계산하여 답변의 신뢰도를 평가합니다.
        
        Args:
            query: 검색 쿼리 (사용자 질문)
            chat_history: 대화 히스토리 (선택적, 컨텍스트 제공)
            top_k: 검색할 문서 수 (기본값: 5, 최대: 10, Bedrock KB 제한)
                - 실제 검색 시 min(top_k, 10)으로 제한됨
                - 호출자가 비용과 성능을 제어할 수 있음
            use_claude: Claude Sonnet 사용 여부 (기본값: True)
            settings: 추가 설정 (선택적)
            confidence_method: 신뢰도 계산 방식 (기본값: "reranker")
                - "reranker": Cross-Encoder Reranker 사용
        
        Returns:
            (has_answer, answer, sources, confidence, doc_number_to_filename, related_queries) 튜플
            - has_answer: 답변 생성 여부 (bool)
            - answer: 생성된 답변 (str 또는 None)
            - sources: 문서 출처 리스트 (list 또는 None)
            - confidence: 신뢰도 점수 (0.0 ~ 1.0)
            - doc_number_to_filename: 문서 번호 → 파일명 매핑 (dict)
            - related_queries: 연관 검색어 리스트 (relevance_score >= 0.7인 문서들에서 추출)
            - primary_category: 주 카테고리 (str 또는 None)
        """
        # 히스토리를 포함한 쿼리 생성
        # 보안: chat_history는 이미 messages.py에서 검증되어 Redis에서 가져온 신뢰할 수 있는 히스토리만 포함
        # 보안: assistant 메시지도 chat_history_validator에서 블랙리스트 검증을 거쳤으므로 신뢰 가능
        # 맥락: 대화 맥락을 유지하기 위해 user와 assistant 메시지를 모두 사용 (대화 쌍으로 구성)
        # 필터링: 히스토리 메시지에 intent_type 메타데이터가 있으면 활용하여 비즈니스 관련 메시지만 포함
        conversation_context = []  # 초기화 (에러 방지)
        if chat_history:
            # 최근 대화 맥락 추출 (user-assistant 쌍으로 구성하여 맥락 유지)
            for msg in chat_history[-4:]:  # 최근 2턴 (user-assistant 쌍)
                # 딕셔너리 형태인 경우
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    intent_type = msg.get('intent_type')  # 히스토리에 저장된 intent_type 확인
                # LangChain 메시지 객체인 경우
                elif isinstance(msg, BaseMessage):
                    if isinstance(msg, HumanMessage):
                        role = 'user'
                    elif isinstance(msg, AIMessage):
                        role = 'assistant'
                    else:
                        role = 'user'
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    # LangChain 메시지 객체는 메타데이터에서 intent_type 확인
                    intent_type = None
                    if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                        intent_type = msg.additional_kwargs.get('intent_type')
                else:
                    role = 'user'
                    content = str(msg)
                    intent_type = None
                
                # intent_type이 있으면 비즈니스 메시지만 포함
                should_include = True
                if intent_type is not None:
                    # intent_type이 "casual"이면 제외 (비즈니스 관련 메시지만 포함)
                    should_include = (intent_type == "business")
                
                if content.strip() and should_include:
                    if role == 'user':
                        conversation_context.append(f"사용자: {content}")
                    elif role == 'assistant':
                        # assistant 메시지도 포함하여 맥락 유지
                        conversation_context.append(f"봇: {content}")
            
            # 대화 맥락이 있으면 추가, 없으면 현재 질문만 사용
            if conversation_context:
                context = "\n".join(conversation_context)
                enhanced_query = f"대화 맥락:\n{context}\n\n현재 질문: {query}"
            else:
                enhanced_query = query
        else:
            enhanced_query = query
        
        # 검색 결과 수: top_k 파라미터 사용 (Bedrock KB 최대값 10 고려)
        search_count = min(top_k, 10)  # Bedrock KB 최대값 제한
        
        # ✅ Bedrock KB 검색에도 enhanced_query(대화 맥락 포함) 사용
        #    - "그거 얼마야?" 같은 대명사 포함 질문이나 짧은 질문의 경우
        #    - 이전 대화 맥락이 없으면 검색 정확도가 매우 떨어짐
        #    - 따라서 대화 맥락이 포함된 쿼리를 사용하여 검색 품질 향상
        bedrock_query = enhanced_query
        
        docs_with_bedrock = self.retriever.similarity_search_with_score(
            bedrock_query,
            k=search_count,
        )
        
        if not docs_with_bedrock:
            return False, None, None, 0.0, {}, [], None
        
        # Reranker 초기화 (지연 로딩)
        if self.reranker is None:
            try:
                # 환경변수 또는 .secrets.toml에서 reranker 타입 선택 (기본값: cross_encoder)
                bedrock_config = get_bedrock_config()
                reranker_type = (bedrock_config.get("reranker_type") or os.getenv("RERANKER_TYPE", "cross_encoder")).lower()
                
                if reranker_type == "bedrock":
                    # Bedrock Rerank 사용
                    model_id = bedrock_config.get("reranker_model_id") or os.getenv("BEDROCK_RERANK_MODEL_ID", "amazon.rerank-v1:0")
                    # Security Fix: 하드코딩된 리전 제거, 설정 없을 경우 ValueError
                    region = bedrock_config.get("region")
                    if not region:
                        raise ValueError("AWS 리전이 설정되지 않았습니다. 환경변수 BEDROCK_REGION 또는 .secrets.toml의 [bedrock].region을 설정하세요.")
                    
                    self.reranker = BedrockReranker(
                        model_id=model_id,
                        region=region,
                        aws_access_key_id=bedrock_config.get("access_key_id"),
                        aws_secret_access_key=bedrock_config.get("secret_access_key"),
                        top_n=10
                    )
                    logger.info(f"Bedrock Reranker 초기화 완료: model_id={model_id}, region={region}")
                else:
                    # Cross-Encoder 사용 (기본값)
                    self.reranker = CrossEncoderReranker()
                    logger.info("Cross-Encoder Reranker 초기화 완료")
            except Exception as e:
                logger.error(f"Reranker 초기화 실패: {e}")
                raise RuntimeError(f"Reranker 초기화에 실패했습니다: {e}") from e
        
        # Reranker로 재정렬
        # Bedrock KB가 반환한 모든 문서를 Reranker로 재정렬
        # top_k를 None으로 설정하여 모든 문서를 재정렬 (더 정확한 순위 결정)
        # ✅ Reranker는 현재 질문만 사용 (짧고 명확한 질문에 최적화)
        #    - Bedrock KB와 Reranker 모두 현재 질문만 사용 (성능 최적화)
        #    - LLM에는 enhanced_query(대화 맥락 포함)가 전달되어 멀티턴 맥락 유지
        rerank_query = extract_user_question(enhanced_query)
        logger.debug(f"Reranker 쿼리 추출: 원본='{enhanced_query[:50]}...' → 추출='{rerank_query[:50]}...'")
        
        docs_with_reranker = self.reranker.rerank_with_original_scores(
            rerank_query,  # 현재 질문만 사용 (Reranker 최적화)
            docs_with_bedrock,
            top_k=None  # 모든 문서 재정렬 (더 정확한 순위)
        )
        
        # ✅ Reranker 점수는 모델별로 스케일이 다르므로 절대값 임계값을 두지 않는다.
        #    - Cohere Rerank 3.5: 0~1 정규화 점수 (1에 가까울수록 관련성 높음)
        #    - Amazon Rerank 1.0: logit 기반 점수 (범위 고정 X, 상대 비교용)
        # 따라서, "max_score < 1.0 이면 비정상" 같은 로직은 제거하고
        # 항상 Reranker 점수를 사용하여 상대적인 순위만 활용한다.
        
        # 한 번의 순회로 모든 필요한 데이터 추출 (성능 최적화)
        docs_with_scores: list[tuple] = []
        reranker_scores = []
        bedrock_scores = []
        
        for doc, bedrock_score, reranker_score in docs_with_reranker:
            reranker_scores.append(reranker_score)
            bedrock_scores.append(bedrock_score)
        
        max_reranker_score = max(reranker_scores) if reranker_scores else 0.0
        min_reranker_score = min(reranker_scores) if reranker_scores else 0.0

        if not reranker_scores:
            # 방어적 폴백: Reranker가 아무 점수도 주지 못한 경우 Bedrock 점수 사용
            logger.warning(
                "⚠️ Reranker 점수가 비어 있어 Bedrock KB 점수를 사용합니다."
            )
            docs_with_scores = [
                (doc, bedrock_score)
                for doc, bedrock_score, _ in docs_with_reranker
            ]
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            # Reranker 점수 사용 (모델별 스케일 그대로, 순위용)
            docs_with_scores = [
                (doc, reranker_score)
                for doc, bedrock_score, reranker_score in docs_with_reranker
            ]
            logger.info(
                "Reranker 점수 사용: 최고=%.6f, 최저=%.6f, 범위=%.6f",
                max_reranker_score,
                min_reranker_score,
                max_reranker_score - min_reranker_score,
            )
        
        first_score = docs_with_scores[0][1] if docs_with_scores else 0.0
        top_score = first_score
        
        # 점수 분포 로깅 (비교용)
        scores = [score for _, score in docs_with_scores]
        max_score = max(scores)
        min_score = min(scores)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        log_msg = (
            f"RAG confidence (reranker): 최종 점수={first_score:.3f}, "
            f"최고={max_score:.3f}, 최저={min_score:.3f}, 평균={avg_score:.3f}, "
            f"결과 수={len(scores)}"
        )
        
        # Bedrock 점수와 비교 로깅
        if bedrock_scores:
            bedrock_max = max(bedrock_scores)
            log_msg += f", Bedrock 최고={bedrock_max:.3f}"
        if reranker_scores:
            reranker_max = max(reranker_scores)
            log_msg += f", Reranker 최고={reranker_max:.3f}"
            # Reranker가 Bedrock보다 개선했는지 확인
            if bedrock_scores:
                improvement = reranker_max - bedrock_max
                if improvement > 0:
                    log_msg += f", 개선={improvement:+.3f}"
                else:
                    log_msg += f", 개선 없음 (Bedrock 사용)"
        
        logger.info(log_msg)
        
        # Reranker 점수는 상대 비교로 처리 (절대값 임계값 사용 안 함)
        # Reranker 점수는 모델 내부 기준에 따라 산출되며, 절대값으로 threshold 비교하면 실패
        # 대신 상대적으로 가장 높은 점수를 가진 문서를 사용
        # confidence는 최고 점수를 그대로 사용 (상대 비교 결과)
        
        # 최소 점수 체크는 제거 (상대 비교만 사용)
        # 만약 모든 문서의 점수가 매우 낮다면, 이미 Bedrock KB에서 필터링되었을 것
        # Reranker는 이미 검색된 문서를 재정렬하는 역할이므로, 최소 임계값 체크는 불필요
        
        # 컨텍스트 생성 (중복 문서 제거하여 다양성 확보)
        # 같은 문서에서 나온 청크가 많으면 다른 문서의 기회를 박탈하므로 제한
        seen_sources = {}
        context_docs = []
        source_to_doc_number = {}  # source -> 문서 번호 매핑 (같은 문서는 같은 번호)
        next_doc_number = 1
        max_chunks_per_source = 3  # 같은 문서에서 최대 3개 청크만 사용
        
        for doc, score in docs_with_scores:
            source = doc.metadata.get('source', 'unknown')
            # 소스별 청크 수 확인
            source_count = seen_sources.get(source, 0)
            if source_count < max_chunks_per_source:
                # 같은 source는 같은 문서 번호 할당
                if source not in source_to_doc_number:
                    source_to_doc_number[source] = next_doc_number
                    next_doc_number += 1
                
                doc_number = source_to_doc_number[source]
                context_docs.append((doc, score, doc_number))
                seen_sources[source] = source_count + 1
        
        # 토큰 제한 적용 (Claude 3.5 Sonnet 기준 200k, 안전하게 180k 설정)
        # 시스템 프롬프트 + 사용자 질문 + 여유분 제외하고 문서에 할당
        MAX_TOTAL_TOKENS = 180000
        SYSTEM_PROMPT_TOKENS = 2000  # 대략적인 추정
        USER_QUERY_TOKENS = count_tokens(enhanced_query) + 500  # 질문 + 여유분
        
        available_context_tokens = MAX_TOTAL_TOKENS - SYSTEM_PROMPT_TOKENS - USER_QUERY_TOKENS
        if available_context_tokens < 1000:
            available_context_tokens = 1000  # 최소 보장
            
        logger.info(f"Token limit check: available={available_context_tokens}, query={USER_QUERY_TOKENS}")

        # 문서 Truncation 적용
        truncated_context_docs = truncate_documents(context_docs, available_context_tokens)
        
        if len(truncated_context_docs) < len(context_docs):
            logger.warning(f"Documents truncated: {len(context_docs)} -> {len(truncated_context_docs)}")
            
        # 컨텍스트 생성 (문서 번호는 source 기반으로 일관되게 할당)
        context = "\n\n".join([
            f"[문서 {doc_number}] {doc.page_content}\n출처: {doc.metadata.get('source', 'unknown')}"
            for doc, score, doc_number in truncated_context_docs
        ])
        
        logger.info(f"RAG 검색: 총 {len(docs_with_scores)}개 결과 중 {len(context_docs)}개 사용 (소스 다양성 확보, 소스 수: {len(seen_sources)})")
        
        # LLM 생성 (설정 기반)
        # Perf: 기본 모델을 claude-haiku-4-5로 변경 (응답 속도 최적화)
        # 사용자가 명시적으로 모델을 선택한 경우 그 선택을 존중
        if settings:
            # Perf: 기본값만 haiku로 설정 (사용자 선택 존중)
            llm_model = settings.get("llm_model", "anthropic.claude-haiku-4-5-20251001-v1:0")
            temperature = settings.get("temperature", 0.7)
            max_tokens = settings.get("max_tokens", 2000)
            persona_description = settings.get("persona_description", "")
            response_length = settings.get("response_length", "normal")
            
            # max_tokens가 너무 작으면 경고
            if max_tokens < 1000:
                logger.warning(
                    f"⚠️ max_tokens={max_tokens}이(가) 너무 작습니다. "
                    f"긴 답변이 중간에 끊길 수 있습니다. 최소 2000 이상 권장합니다."
                )
            
            # LLM 생성 (설정값 적용)
            llm = self.provider.get_chat_llm(
                role="business",
                model_name=llm_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            model_name = llm_model
        else:
            # Perf: 기본 LLM을 haiku로 변경 (응답 속도 최적화)
            # business 모델 대신 casual 모델(haiku) 사용
            llm = self.provider.get_chat_llm(role="casual")
            try:
                model_name = getattr(llm, "model_name", "claude-haiku-4-5")
            except Exception:
                model_name = "claude-haiku-4-5"
        
        # max_tokens 값 가져오기 (토큰 제한 안내용)
        current_max_tokens = max_tokens if settings else 2000
        
        # 마크다운 허용 여부에 따른 답변 형식 규칙
        if response_format == "markdown":
            format_rules = f"""**답변 본문 작성 규칙 (Markdown 허용):**
- Markdown 문법을 사용하여 가독성 높은 답변을 작성하세요.
- **굵게**, *기울임*, `코드`, 리스트(-, *), 제목(##) 등을 활용하세요.
- 본문에 출처를 표기하지 마세요. 출처는 document_numbers에만 포함하세요.
- 구조화된 정보는 리스트나 표로 표현하세요.

**토큰 제한 (매우 중요):**
- 최대 토큰 수: {current_max_tokens} 토큰
- 반드시 이 제한 내에서 완전한 JSON 응답을 생성하세요.
- JSON 형식이 완성되지 않으면 응답이 무효화됩니다.
- 토큰이 부족할 수 있으므로 핵심 정보만 간결하게 작성하세요."""
        else:
            format_rules = f"""**답변 본문 작성 규칙 (매우 중요):**
- 절대 Markdown 형식을 사용하지 마세요. **굵게**, *기울임*, `코드`, [], (), # 제목, - 리스트 등 모든 Markdown 문법을 사용하지 마세요.
- 일반 텍스트로만 작성하세요. 숫자, 문장 부호, 한글, 영문만 사용하세요.
- 본문에 출처를 표기하지 마세요. 출처는 document_numbers에만 포함하세요.
- 자연스럽고 읽기 쉬운 일반 텍스트 형식으로 작성하세요.
- 예시: "카카오 i 커넥트 메시지는 기업이 고객에게 문자, 카카오톡, RCS 등 다양한 디지털 채널로 메시지를 발송할 수 있는 플랫폼입니다."

**응답 길이 제한 (매우 중요):**
- 답변 본문(answer)은 반드시 500자 이내로 작성하세요.
- 카카오 워크 blockit 텍스트 제한으로 인해 500자를 초과하면 답변이 표시되지 않습니다.
- 핵심 정보만 간결하게 전달하세요.

**토큰 제한 (매우 중요):**
- 최대 토큰 수: {current_max_tokens} 토큰
- 반드시 이 제한 내에서 완전한 JSON 응답을 생성하세요.
- JSON 형식이 완성되지 않으면 응답이 무효화됩니다.
- 토큰이 부족할 수 있으므로 핵심 정보만 간결하게 작성하세요."""
        
        # 시스템 프롬프트 (페르소나 설정 적용)
        if settings and settings.get("persona_description"):
            base_persona = settings["persona_description"]
            response_length = settings.get("response_length", "normal")
            system_prompt = f"""{base_persona}

**추가 규칙:**
1. 제공된 문서의 내용만을 바탕으로 답변하세요.
2. 문서에 없는 내용은 추측하거나 만들지 마세요.
3. 명확하지 않으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

**응답 형식 (매우 중요):**
반드시 다음 JSON 형식으로만 응답하세요. 다른 형식은 사용하지 마세요.
**중요: JSON 응답은 반드시 완전하게 생성되어야 합니다. 중간에 끊기면 안 됩니다.**
**토큰 제한({current_max_tokens} 토큰) 내에서 반드시 완전한 JSON을 생성하세요.**

{{
  "answer": "답변 본문",
  "document_numbers": [1, 2, 3],
  "related_queries": ["연관 검색어 1", "연관 검색어 2", "연관 검색어 3"]
}}

{format_rules}

**related_queries 작성 규칙:**
- 답변 내용 및 문서와 관련된 추가 질문이나 검색어를  3개 작성하세요.
- 사용자가 다음에 궁금해할 만한 내용을 추천하세요.
- 짧고 간결하게 작성하세요.

**document_numbers 규칙:**
- 실제로 답변에 사용한 문서 번호만 포함하세요.
- 배열 형식으로 작성하세요 (예: [1] 또는 [1, 2, 3]).
- 사용하지 않은 문서 번호는 포함하지 마세요.

**응답 길이:** {response_length}"""
        else:
            system_prompt = f"""당신은 사내 문서 기반 Q&A 어시스턴트입니다.

**규칙:**
1. 제공된 문서의 내용만을 바탕으로 답변하세요.
2. 문서에 없는 내용은 추측하거나 만들지 마세요.
3. 명확하지 않으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

**응답 형식 (매우 중요):**
반드시 다음 JSON 형식으로만 응답하세요. 다른 형식은 사용하지 마세요.
**중요: JSON 응답은 반드시 완전하게 생성되어야 합니다. 중간에 끊기면 안 됩니다.**
**토큰 제한({current_max_tokens} 토큰) 내에서 반드시 완전한 JSON을 생성하세요.**

{{
  "answer": "답변 본문",
  "document_numbers": [1, 2, 3],
  "related_queries": ["연관 검색어 1", "연관 검색어 2", "연관 검색어 3"]
}}

{format_rules}

**related_queries 작성 규칙:**
- 답변 내용 및 문서와 관련된 추가 질문이나 검색어를 3개 작성하세요.
- 사용자가 다음에 궁금해할 만한 내용을 추천하세요.
- 짧고 간결하게 작성하세요.

**document_numbers 규칙:**
- 실제로 답변에 사용한 문서 번호만 포함하세요.
- 배열 형식으로 작성하세요 (예: [1] 또는 [1, 2, 3]).
- 사용하지 않은 문서 번호는 포함하지 마세요."""
        # 멀티턴 지원: enhanced_query 사용 (대화 맥락 포함)
        # enhanced_query에는 이전 대화 맥락이 포함되어 있으므로 이를 LLM에 전달
        user_prompt = f"""참고 문서:

{context}

{enhanced_query}

위 질문에 대해 JSON 형식으로 답변하세요."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Circuit Breaker를 통해 LLM 호출 (장애 복구)
            # LLMProvider의 circuit_breaker 사용
            if hasattr(self, 'provider') and hasattr(self.provider, 'invoke_with_resilience'):
                response = self.provider.invoke_with_resilience(llm, messages)
            else:
                logger.warning("LLMProvider의 invoke_with_resilience를 사용할 수 없습니다. 직접 호출합니다.")
                response = llm.invoke(messages)
            raw_response = response.content.strip()
            
            # JSON 파싱 (구조화된 출력)
            import json
            
            # JSON 파싱 및 마크다운 제거 (rag_utils 사용)
            answer, document_numbers, related_queries = parse_json_response(
                raw_response,
                response_format
            )
            
            # 문서 번호와 파일명 매핑 생성 (컨텍스트에 사용된 source_to_doc_number 기반)
            # 컨텍스트 생성 시 이미 source_to_doc_number 매핑이 생성되었으므로 이를 활용
            doc_number_to_filename = {}
            unique_sources = {}  # 중복 제거를 위한 딕셔너리 (source -> filename)
            
            # source -> doc 매핑 생성 (메타데이터에서 파일명 추출을 위해)
            source_to_doc = {}
            for doc, score, doc_number in context_docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_to_doc:
                    source_to_doc[source] = doc
            
            # 컨텍스트에 사용된 모든 source에 대해 파일명 추출 및 매핑 생성
            for source, doc_number in source_to_doc_number.items():
                filename = None
                
                # 1. 메타데이터에서 정제된 파일명 추출 (우선순위 최우선)
                if source in source_to_doc:
                    doc = source_to_doc[source]
                    # display_name은 bedrock_kb.py에서 메타데이터의 title/filename/name을 추출한 것
                    filename = doc.metadata.get('display_name')
                    if filename:
                        logger.debug(f"파일명을 메타데이터에서 추출: {filename} (source: {source})")
                
                # 메타데이터에 없으면 URI에서 추출
                if not filename:
                    filename = extract_filename_from_uri(source)
                    if filename != source:
                        logger.debug(f"파일명을 URI에서 추출: {filename} (source: {source})")
                    else:
                        logger.debug(f"파일명 추출 실패, 원본 사용: {source}")
                
                # 문서 번호와 파일명 매핑 (컨텍스트에 사용된 번호 그대로 사용)
                doc_number_to_filename[doc_number] = filename
                
                # 중복 제거하여 sources 리스트 생성 (파일명만 저장)
                if source not in unique_sources:
                    unique_sources[source] = filename
            
            # 출처 리스트 생성 (LLM이 지정한 document_numbers에 해당하는 파일명만)
            if document_numbers:
                # LLM이 지정한 문서 번호에 해당하는 파일명만 사용
                sources = [
                    doc_number_to_filename[doc_num] 
                    for doc_num in document_numbers 
                    if doc_num in doc_number_to_filename
                ]
                # 중복 제거
                seen = set()
                unique_sources_list = []
                for s in sources:
                    if s and s.lower() not in seen:
                        unique_sources_list.append(s)
                        seen.add(s.lower())
                sources = unique_sources_list
            else:
                # document_numbers가 없으면 모든 사용된 문서 사용 (fallback)
                sources = list(unique_sources.values())
            
            # 연관 검색어 추출 (LLM 생성 결과 사용)
            # related_queries는 이미 JSON 파싱 단계에서 추출됨
            if not related_queries:
                related_queries = []
            
            # 최대 3개로 제한
            related_queries = related_queries[:3]
            
            # 디버깅: 매핑 정보 로깅
            logger.info(f"RAG answer generated using {model_name}")
            logger.info(f"Document number mapping: {doc_number_to_filename}")
            logger.info(f"LLM specified document_numbers: {document_numbers}")
            logger.info(f"Final sources: {sources}")
            logger.info(f"Related queries (LLM generated): {related_queries}")
            logger.info(f"Answer (first 200 chars): {answer[:200]}")
            
            # 주 카테고리 추출 (가장 높은 점수의 문서에서 추출)
            primary_category = None
            if docs_with_scores:
                logger.info(f"🔍 [Category] Searching in {len(docs_with_scores)} documents for category")
                # 상위 문서들에서 카테고리 찾기 (최대 3개까지 확인)
                for idx, (doc, score) in enumerate(docs_with_scores[:3]):
                    metadata = doc.metadata
                    logger.info(f"🔍 [Category] Doc {idx+1} metadata keys: {list(metadata.keys())}")
                    # 다양한 필드명 시도 (대소문자 구분 없이)
                    primary_category = (
                        metadata.get('category') or 
                        metadata.get('Category') or 
                        metadata.get('CATEGORY') or
                        metadata.get('category_name') or
                        metadata.get('categoryName') or
                        metadata.get('document_category') or
                        metadata.get('DocumentCategory')
                    )
                    
                    # 메타데이터에 카테고리가 없으면 S3 경로에서 추출 시도
                    # 경로 패턴: s3://bucket/{category_id}/filename.pdf
                    if not primary_category:
                        source = metadata.get('source', '') or metadata.get('x-amz-bedrock-kb-source-uri', '')
                        if source and source.startswith('s3://'):
                            # s3://bucket/category_id/filename.pdf 형태에서 category_id 추출
                            try:
                                # s3:// 제거 후 경로 분리
                                path_without_scheme = source.replace('s3://', '')
                                parts = path_without_scheme.split('/')
                                logger.debug(f"🔍 [Category] S3 path parts: {parts}")
                                
                                if len(parts) >= 2:
                                    # bucket 다음이 카테고리 ID일 가능성이 높음
                                    potential_category = parts[1]
                                    logger.debug(f"🔍 [Category] Potential category from path: '{potential_category}'")
                                    
                                    # 숫자로만 구성된 경우 카테고리 ID로 간주
                                    if potential_category.isdigit():
                                        primary_category = potential_category
                                        logger.info(f"✅ Category extracted from S3 path (async): {primary_category} from {source}")
                                    else:
                                        logger.debug(f"🔍 [Category] Path segment '{potential_category}' is not numeric, skipping")
                                else:
                                    logger.debug(f"🔍 [Category] S3 path has insufficient parts: {len(parts)} < 2")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to extract category from S3 path {source}: {e}")
                    
                    if primary_category:
                        logger.info(f"✅ Primary category extracted: {primary_category} from doc {idx+1} with score={score:.4f}")
                        break
                
                # 카테고리를 찾지 못한 경우 메타데이터 로깅 (디버깅용)
                if not primary_category and docs_with_scores:
                    top_doc = docs_with_scores[0][0]
                    top_source = top_doc.metadata.get('source', '') or top_doc.metadata.get('x-amz-bedrock-kb-source-uri', '')
                    logger.warning(f"⚠️ Category not found. Source: {top_source}, Available keys: {list(top_doc.metadata.keys())}")
            else:
                logger.warning("⚠️ docs_with_scores is empty, cannot extract category")
            
            return True, answer, sources, top_score, doc_number_to_filename, related_queries, primary_category
            
        except Exception as e:
            return False, None, None, 0.0, {}, [], None

    async def asearch(
        self, 
        query: str, 
        chat_history: list = None,
        top_k: int = 5,
        use_claude: bool = True,
        settings: Optional[dict] = None,
        confidence_method: Literal["reranker"] = "reranker",
        response_format: str = "plain"
    ) -> Tuple[bool, Optional[str], Optional[list], float, Dict[int, str], list, Optional[str]]:
        """RAG 비동기 검색 및 답변 생성"""
        # 히스토리를 포함한 쿼리 생성 (동기 로직 재사용)
        conversation_context = []
        if chat_history:
            for msg in chat_history[-4:]:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    intent_type = msg.get('intent_type')
                elif isinstance(msg, BaseMessage):
                    if isinstance(msg, HumanMessage):
                        role = 'user'
                    elif isinstance(msg, AIMessage):
                        role = 'assistant'
                    else:
                        role = 'user'
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    intent_type = None
                    if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                        intent_type = msg.additional_kwargs.get('intent_type')
                else:
                    role = 'user'
                    content = str(msg)
                    intent_type = None
                
                should_include = True
                if intent_type is not None:
                    should_include = (intent_type == "business")
                
                if content.strip() and should_include:
                    if role == 'user':
                        conversation_context.append(f"사용자: {content}")
                    elif role == 'assistant':
                        conversation_context.append(f"봇: {content}")
            
            if conversation_context:
                context = "\n".join(conversation_context)
                enhanced_query = f"대화 맥락:\n{context}\n\n현재 질문: {query}"
            else:
                enhanced_query = query
        else:
            enhanced_query = query
        
        search_count = min(top_k, 10)
        bedrock_query = enhanced_query
        
        # 비동기 검색 실행
        docs_with_bedrock = await self.retriever.asimilarity_search_with_score(
            bedrock_query,
            k=search_count,
        )
        
        if not docs_with_bedrock:
            return False, None, None, 0.0, {}, [], None
        
        # Reranker 초기화 (지연 로딩)
        if self.reranker is None:
            try:
                bedrock_config = get_bedrock_config()
                reranker_type = (bedrock_config.get("reranker_type") or os.getenv("RERANKER_TYPE", "cross_encoder")).lower()
                
                if reranker_type == "bedrock":
                    model_id = bedrock_config.get("reranker_model_id") or os.getenv("BEDROCK_RERANK_MODEL_ID", "amazon.rerank-v1:0")
                    # Security Fix: 하드코딩된 리전 제거, 설정 없을 경우 ValueError
                    region = bedrock_config.get("region")
                    if not region:
                        raise ValueError("AWS 리전이 설정되지 않았습니다. 환경변수 BEDROCK_REGION 또는 .secrets.toml의 [bedrock].region을 설정하세요.")
                    
                    self.reranker = BedrockReranker(
                        model_id=model_id,
                        region=region,
                        aws_access_key_id=bedrock_config.get("access_key_id"),
                        aws_secret_access_key=bedrock_config.get("secret_access_key"),
                        top_n=10
                    )
                    logger.info(f"Bedrock Reranker 초기화 완료: model_id={model_id}, region={region}")
                else:
                    self.reranker = CrossEncoderReranker()
                    logger.info("Cross-Encoder Reranker 초기화 완료")
            except Exception as e:
                logger.error(f"Reranker 초기화 실패: {e}")
                raise RuntimeError(f"Reranker 초기화에 실패했습니다: {e}") from e
        
        # 1차 검색 결과 평가 (Reranker 사용)
        max_score = 0.0
        docs_with_reranker = []
        
        if docs_with_bedrock:
            rerank_query = extract_user_question(enhanced_query)
            docs_with_reranker = await self.reranker.arerank_with_original_scores(
                rerank_query,
                docs_with_bedrock,
                top_k=None
            )
            if docs_with_reranker:
                # Reranker 점수 확인 (세 번째 요소가 reranker_score)
                scores = [score for _, _, score in docs_with_reranker]
                max_score = max(scores) if scores else 0.0
        
        logger.info(f"1차 검색 결과: query='{enhanced_query}', docs={len(docs_with_bedrock)}, max_score={max_score:.4f}")
        
        # 2. 조건부 검색어 최적화 (Conditional Query Refinement)
        # 점수가 낮거나 결과가 없으면 검색어 최적화 시도
        # 임계값 0.45: "비즈메세지"(0.25) vs "비즈 메시지"(0.36~0.56) 고려하여 상향 조정
        REFINE_THRESHOLD = 0.45
        
        if not docs_with_bedrock or max_score < REFINE_THRESHOLD:
            logger.info(f"검색 품질 저하 감지 (score={max_score:.4f} < {REFINE_THRESHOLD}). 검색어 최적화 시도...")
            
            refined_query = await self._refine_query(enhanced_query)
            
            if refined_query != enhanced_query:
                # 최적화된 쿼리로 2차 검색
                bedrock_query = refined_query
                docs_with_bedrock_2nd = await self.retriever.asimilarity_search_with_score(
                    bedrock_query,
                    k=search_count,
                )
                
                if docs_with_bedrock_2nd:
                    # 2차 검색 결과 재정렬
                    rerank_query_2nd = extract_user_question(refined_query)
                    docs_with_reranker_2nd = await self.reranker.arerank_with_original_scores(
                        rerank_query_2nd,
                        docs_with_bedrock_2nd,
                        top_k=None
                    )
                    
                    # 2차 검색 점수 확인
                    scores_2nd = [score for _, _, score in docs_with_reranker_2nd]
                    max_score_2nd = max(scores_2nd) if scores_2nd else 0.0
                    
                    logger.info(f"2차 검색 결과: query='{refined_query}', docs={len(docs_with_bedrock_2nd)}, max_score={max_score_2nd:.4f}")
                    
                    # 2차 검색 결과가 더 좋거나, 1차 결과가 아예 없으면 교체
                    if not docs_with_bedrock or max_score_2nd > max_score:
                        logger.info(f"검색어 최적화 효과 확인: 점수 개선 ({max_score:.4f} -> {max_score_2nd:.4f}). 2차 결과 사용.")
                        docs_with_bedrock = docs_with_bedrock_2nd
                        docs_with_reranker = docs_with_reranker_2nd
                        # 주의: enhanced_query는 LLM 생성 시 사용되므로 원본 유지 또는 교체 고려
                        # 여기서는 검색 결과만 교체하고, LLM에는 원본 질문 맥락을 주는 것이 맞을 수 있으나
                        # 오타가 교정된 쿼리가 더 정확한 맥락일 수 있음.
                        # 하지만 enhanced_query는 대화 맥락이 포함된 것이라 단순 교체 어려움.
                        # 검색 결과(docs)가 바뀌었으므로 충분함.
                    else:
                        logger.info("검색어 최적화 효과 미미. 1차 결과 유지.")
                else:
                    logger.info("2차 검색 결과 없음. 1차 결과 유지.")
            else:
                logger.info("검색어 최적화 결과 변경 없음.")
        else:
            logger.info(f"검색 품질 양호 (score={max_score:.4f} >= {REFINE_THRESHOLD}). 검색어 최적화 스킵.")

        if not docs_with_bedrock:
            return False, None, None, 0.0, {}, [], None
            
        # 점수 처리 로직 (이미 docs_with_reranker가 계산되었으므로 재사용)
        
        # 점수 처리 로직 (동기 코드와 동일, 성능 최적화)
        docs_with_scores: list[tuple] = []
        reranker_scores = []
        bedrock_scores = []
        
        for doc, bedrock_score, reranker_score in docs_with_reranker:
            reranker_scores.append(reranker_score)
            bedrock_scores.append(bedrock_score)
        
        max_reranker_score = max(reranker_scores) if reranker_scores else 0.0
        min_reranker_score = min(reranker_scores) if reranker_scores else 0.0

        if not reranker_scores:
            logger.warning("⚠️ Reranker 점수가 비어 있어 Bedrock KB 점수를 사용합니다.")
            docs_with_scores = [
                (doc, bedrock_score)
                for doc, bedrock_score, _ in docs_with_reranker
            ]
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            docs_with_scores = [
                (doc, reranker_score)
                for doc, bedrock_score, reranker_score in docs_with_reranker
            ]
            logger.info(
                "Reranker 점수 사용: 최고=%.6f, 최저=%.6f, 범위=%.6f",
                max_reranker_score,
                min_reranker_score,
                max_reranker_score - min_reranker_score,
            )
        
        first_score = docs_with_scores[0][1] if docs_with_scores else 0.0
        top_score = first_score
        
        # 컨텍스트 생성
        seen_sources = {}
        context_docs = []
        source_to_doc_number = {}
        next_doc_number = 1
        max_chunks_per_source = 3
        
        for doc, score in docs_with_scores:
            source = doc.metadata.get('source', 'unknown')
            source_count = seen_sources.get(source, 0)
            if source_count < max_chunks_per_source:
                if source not in source_to_doc_number:
                    source_to_doc_number[source] = next_doc_number
                    next_doc_number += 1
                
                doc_number = source_to_doc_number[source]
                context_docs.append((doc, score, doc_number))
                seen_sources[source] = source_count + 1
        
        # 토큰 제한 적용 (비동기)
        MAX_TOTAL_TOKENS = 180000
        SYSTEM_PROMPT_TOKENS = 2000
        USER_QUERY_TOKENS = count_tokens(enhanced_query) + 500
        
        available_context_tokens = MAX_TOTAL_TOKENS - SYSTEM_PROMPT_TOKENS - USER_QUERY_TOKENS
        if available_context_tokens < 1000:
            available_context_tokens = 1000
            
        logger.info(f"Token limit check (async): available={available_context_tokens}, query={USER_QUERY_TOKENS}")

        truncated_context_docs = truncate_documents(context_docs, available_context_tokens)
        
        if len(truncated_context_docs) < len(context_docs):
            logger.warning(f"Documents truncated (async): {len(context_docs)} -> {len(truncated_context_docs)}")

        context = "\n\n".join([
            f"[문서 {doc_number}] {doc.page_content}\n출처: {doc.metadata.get('source', 'unknown')}"
            for doc, score, doc_number in truncated_context_docs
        ])
        
        logger.info(f"RAG 검색: 총 {len(docs_with_scores)}개 결과 중 {len(context_docs)}개 사용")
        
        # LLM 생성
        # Perf: 기본 모델을 claude-haiku-4-5로 변경 (응답 속도 최적화)
        # 사용자가 명시적으로 모델을 선택한 경우 그 선택을 존중
        if settings:
            # Perf: 기본값만 haiku로 설정 (사용자 선택 존중)
            llm_model = settings.get("llm_model", "anthropic.claude-haiku-4-5-20251001-v1:0")
            temperature = settings.get("temperature", 0.7)
            max_tokens = settings.get("max_tokens", 2000)
            persona_description = settings.get("persona_description", "")
            response_length = settings.get("response_length", "normal")
            
            llm = self.provider.get_chat_llm(
                role="business",
                model_name=llm_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            model_name = llm_model
        else:
            # Perf: 기본 LLM을 haiku로 변경 (응답 속도 최적화)
            # business 모델 대신 casual 모델(haiku) 사용
            llm = self.provider.get_chat_llm(role="casual")
            try:
                model_name = getattr(llm, "model_name", "claude-haiku-4-5")
            except Exception:
                model_name = "claude-haiku-4-5"
        
        # max_tokens 값 가져오기 (토큰 제한 안내용)
        if settings:
            current_max_tokens = max_tokens
        else:
            current_max_tokens = 2000
        
        # 마크다운 허용 여부에 따른 답변 형식 규칙 (search 메서드와 동일)
        if response_format == "markdown":
            format_rules = """**답변 본문 작성 규칙 (Markdown 허용):**
- Markdown 문법을 사용하여 가독성 높은 답변을 작성하세요.
- **굵게**, *기울임*, `코드`, 리스트(-, *), 제목(##) 등을 활용하세요.
- 본문에 출처를 표기하지 마세요. 출처는 document_numbers에만 포함하세요.
- 구조화된 정보는 리스트나 표로 표현하세요."""
        else:
            format_rules = """**답변 본문 작성 규칙 (매우 중요):**
- 절대 Markdown 형식을 사용하지 마세요. **굵게**, *기울임*, `코드`, [], (), # 제목, - 리스트 등 모든 Markdown 문법을 사용하지 마세요.
- 일반 텍스트로만 작성하세요. 숫자, 문장 부호, 한글, 영문만 사용하세요.
- 본문에 출처를 표기하지 마세요. 출처는 document_numbers에만 포함하세요.
- 자연스럽고 읽기 쉬운 일반 텍스트 형식으로 작성하세요.
- 예시: "카카오 i 커넥트 메시지는 기업이 고객에게 문자, 카카오톡, RCS 등 다양한 디지털 채널로 메시지를 발송할 수 있는 플랫폼입니다."

**응답 길이 제한 (매우 중요):**
- 답변 본문(answer)은 반드시 500자 이내로 작성하세요.
- 카카오 워크 blockit 텍스트 제한으로 인해 500자를 초과하면 답변이 표시되지 않습니다.
- 핵심 정보만 간결하게 전달하세요."""
        
        # 프롬프트 구성 (search 메서드와 동일)
        if settings and settings.get("persona_description"):
            base_persona = settings["persona_description"]
            response_length = settings.get("response_length", "normal")
            system_prompt = f"""{base_persona}

**추가 규칙:**
1. 제공된 문서의 내용만을 바탕으로 답변하세요.
2. 문서에 없는 내용은 추측하거나 만들지 마세요.
3. 명확하지 않으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

**응답 형식 (매우 중요):**
반드시 다음 JSON 형식으로만 응답하세요. 다른 형식은 사용하지 마세요.
**중요: JSON 응답은 반드시 완전하게 생성되어야 합니다. 중간에 끊기면 안 됩니다.**
**토큰 제한({current_max_tokens} 토큰) 내에서 반드시 완전한 JSON을 생성하세요.**

{{
  "answer": "답변 본문",
  "document_numbers": [1, 2, 3],
  "related_queries": ["연관 검색어 1", "연관 검색어 2", "연관 검색어 3"]
}}

{format_rules}

**related_queries 작성 규칙:**
- 답변 내용 및 문서와 관련된 추가 질문이나 검색어를 3개 작성하세요.
- 사용자가 다음에 궁금해할 만한 내용을 추천하세요.
- 짧고 간결하게 작성하세요.

**document_numbers 규칙:**
- 실제로 답변에 사용한 문서 번호만 포함하세요.
- 배열 형식으로 작성하세요 (예: [1] 또는 [1, 2, 3]).
- 사용하지 않은 문서 번호는 포함하지 마세요.

**응답 길이:** {response_length}"""
        else:
            system_prompt = f"""당신은 사내 문서 기반 Q&A 어시스턴트입니다.

**규칙:**
1. 제공된 문서의 내용만을 바탕으로 답변하세요.
2. 문서에 없는 내용은 추측하거나 만들지 마세요.
3. 명확하지 않으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

**응답 형식 (매우 중요):**
반드시 다음 JSON 형식으로만 응답하세요. 다른 형식은 사용하지 마세요.
**중요: JSON 응답은 반드시 완전하게 생성되어야 합니다. 중간에 끊기면 안 됩니다.**
**토큰 제한({current_max_tokens} 토큰) 내에서 반드시 완전한 JSON을 생성하세요.**

{{
  "answer": "답변 본문",
  "document_numbers": [1, 2, 3],
  "related_queries": ["연관 검색어 1", "연관 검색어 2", "연관 검색어 3"]
}}

{format_rules}

**related_queries 작성 규칙:**
- 답변 내용 및 문서와 관련된 추가 질문이나 검색어를 3개 작성하세요.
- 사용자가 다음에 궁금해할 만한 내용을 추천하세요.
- 짧고 간결하게 작성하세요.

**document_numbers 규칙:**
- 실제로 답변에 사용한 문서 번호만 포함하세요.
- 배열 형식으로 작성하세요 (예: [1] 또는 [1, 2, 3]).
- 사용하지 않은 문서 번호는 포함하지 마세요."""
        
        user_prompt = f"""참고 문서:

{context}

{enhanced_query}

위 질문에 대해 JSON 형식으로 답변하세요."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # 비동기 LLM 호출
            if hasattr(self, 'provider') and hasattr(self.provider, 'ainvoke_with_resilience'):
                response = await self.provider.ainvoke_with_resilience(llm, messages)
            else:
                logger.warning("LLMProvider의 ainvoke_with_resilience를 사용할 수 없습니다. 직접 호출합니다.")
                if hasattr(llm, "ainvoke"):
                    response = await llm.ainvoke(messages)
                else:
                    import asyncio
                    response = await asyncio.to_thread(llm.invoke, messages)
            
            raw_response = response.content.strip()
            
            # JSON 파싱 및 마크다운 제거 (rag_utils 사용)
            answer, document_numbers, related_queries = parse_json_response(
                raw_response,
                response_format
            )
            
            doc_number_to_filename = {}
            unique_sources = {}
            source_to_doc = {}
            for doc, score, doc_number in context_docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_to_doc:
                    source_to_doc[source] = doc
            
            for source, doc_number in source_to_doc_number.items():
                filename = None
                if source in source_to_doc:
                    doc = source_to_doc[source]
                    filename = doc.metadata.get('display_name')
                
                if not filename:
                    filename = extract_filename_from_uri(source)
                
                doc_number_to_filename[doc_number] = filename
                
                if source not in unique_sources:
                    unique_sources[source] = filename
            
            if document_numbers:
                sources = [
                    doc_number_to_filename[doc_num] 
                    for doc_num in document_numbers 
                    if doc_num in doc_number_to_filename
                ]
                seen = set()
                unique_sources_list = []
                for s in sources:
                    if s and s.lower() not in seen:
                        unique_sources_list.append(s)
                        seen.add(s.lower())
                sources = unique_sources_list
            else:
                sources = list(unique_sources.values())
            
            # 연관 검색어 추출 (LLM 생성 결과 사용)
            # related_queries는 이미 JSON 파싱 단계에서 추출됨
            if not related_queries:
                related_queries = []
            
            # 최대 3개로 제한
            related_queries = related_queries[:3]
            
            logger.info(f"RAG answer generated using {model_name}")
            logger.info(f"Related queries (LLM generated): {related_queries}")
            
            # 주 카테고리 추출 (가장 높은 점수의 문서에서 추출)
            primary_category = None
            if docs_with_scores:
                logger.info(f"🔍 [Category] Searching in {len(docs_with_scores)} documents for category")
                # 상위 문서들에서 카테고리 찾기 (최대 3개까지 확인)
                for idx, (doc, score) in enumerate(docs_with_scores[:3]):
                    metadata = doc.metadata
                    logger.info(f"🔍 [Category] Doc {idx+1} metadata keys: {list(metadata.keys())}")
                    # 다양한 필드명 시도 (대소문자 구분 없이)
                    primary_category = (
                        metadata.get('category') or 
                        metadata.get('Category') or 
                        metadata.get('CATEGORY') or
                        metadata.get('category_name') or
                        metadata.get('categoryName') or
                        metadata.get('document_category') or
                        metadata.get('DocumentCategory')
                    )
                    
                    # 메타데이터에 카테고리가 없으면 S3 경로에서 추출 시도
                    # 경로 패턴: s3://bucket/{category_id}/filename.pdf
                    if not primary_category:
                        source = metadata.get('source', '') or metadata.get('x-amz-bedrock-kb-source-uri', '')
                        if source and source.startswith('s3://'):
                            # s3://bucket/category_id/filename.pdf 형태에서 category_id 추출
                            try:
                                # s3:// 제거 후 경로 분리
                                path_without_scheme = source.replace('s3://', '')
                                parts = path_without_scheme.split('/')
                                logger.debug(f"🔍 [Category] S3 path parts: {parts}")
                                
                                if len(parts) >= 2:
                                    # bucket 다음이 카테고리 ID일 가능성이 높음
                                    potential_category = parts[1]
                                    logger.debug(f"🔍 [Category] Potential category from path: '{potential_category}'")
                                    
                                    # 숫자로만 구성된 경우 카테고리 ID로 간주
                                    if potential_category.isdigit():
                                        primary_category = potential_category
                                        logger.info(f"✅ Category extracted from S3 path (async): {primary_category} from {source}")
                                    else:
                                        logger.debug(f"🔍 [Category] Path segment '{potential_category}' is not numeric, skipping")
                                else:
                                    logger.debug(f"🔍 [Category] S3 path has insufficient parts: {len(parts)} < 2")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to extract category from S3 path {source}: {e}")
                    
                    if primary_category:
                        logger.info(f"✅ Primary category extracted (async): {primary_category} from doc {idx+1} with score={score:.4f}")
                        break
                
                # 카테고리를 찾지 못한 경우 메타데이터 로깅 (디버깅용)
                if not primary_category and docs_with_scores:
                    top_doc = docs_with_scores[0][0]
                    top_source = top_doc.metadata.get('source', '') or top_doc.metadata.get('x-amz-bedrock-kb-source-uri', '')
                    logger.warning(f"⚠️ Category not found. Source: {top_source}, Available keys: {list(top_doc.metadata.keys())}")
            else:
                logger.warning("⚠️ docs_with_scores is empty, cannot extract category")
            
            return True, answer, sources, top_score, doc_number_to_filename, related_queries, primary_category
            
        except Exception as e:
            logger.error(f"RAG LLM error: {e}")
            return False, None, None, 0.0, {}, [], None
    
    def _extract_filename_from_uri(self, uri: str) -> str:
        """URI에서 파일명 추출 (메타데이터가 없을 때 폴백용)
        
        다양한 URI 형식을 지원:
        - s3://bucket/path/file.pdf -> file.pdf
        - s3://bucket/path/file.pdf/paragraphs/v1/xxx.json -> file.pdf
        - file:///path/to/document.pdf -> document.pdf
        - /path/to/document.pdf -> document.pdf
        - http://example.com/file.pdf -> file.pdf
        
        Args:
            uri: 파일 URI 또는 경로
        
        Returns:
            추출된 파일명 또는 원본 URI (추출 실패 시)
        """
        if not uri or not isinstance(uri, str):
            return uri or "unknown"
        
        # 경로 분리
        parts = uri.split("/")
        
        # 파일명 찾기 (확장자가 있는 부분)
        # 지원하는 확장자 목록
        supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".xlsx", ".pptx"]
        
        # 역순으로 검색 (마지막에 있는 파일명 우선)
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if "." in part and any(part.lower().endswith(ext) for ext in supported_extensions):
                return part
        
        # 파일명을 찾지 못하면 마지막 의미있는 부분 반환
        if len(parts) > 0:
            last_part = parts[-1]
            if last_part and last_part != "":
                return last_part
        
        # 그래도 못 찾으면 원본 반환
        return uri
    






    async def _refine_query(self, query: str) -> str:
        """검색어 최적화 (Query Refinement)
        
        사용자 질문의 오타, 띄어쓰기 오류 등을 교정하고
        검색에 적합한 형태로 변환하여 검색 정확도 향상
        
        Args:
            query: 원본 사용자 질문
            
        Returns:
            최적화된 검색어 (변경 없으면 원본 반환)
        """
        # 1. 너무 짧거나 긴 쿼리는 스킵 (비용 절감)
        if len(query) < 2 or len(query) > 100:
            return query
            
        try:
            # 2. LLM을 사용하여 검색어 교정
            # Haiku 모델 사용 (빠르고 저렴함)
            llm = self.provider.get_chat_llm(role="casual")
            
            system_prompt = """당신은 검색어 최적화 도구입니다.
사용자 질문을 분석하여 검색 엔진이 더 잘 이해할 수 있는 형태로 교정하세요.
다음 규칙을 따르세요:
1. 오타 수정 및 띄어쓰기 교정 (예: "비즈메세지" -> "비즈 메세지")
2. 핵심 키워드 유지
3. 불필요한 조사 제거 (선택사항)
4. 결과는 교정된 검색어만 출력 (설명 금지)
5. 교정이 필요 없으면 원본 그대로 출력"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # 비동기 호출
            response = await self.provider.ainvoke_with_resilience(llm, messages)
            refined_query = response.content.strip()
            
            # 3. 결과 검증
            # 너무 길어지거나 이상한 결과는 원본 사용
            if len(refined_query) > len(query) * 2:
                logger.warning(f"Refined query too long: {refined_query} (orig: {query})")
                return query
                
            if refined_query != query:
                logger.info(f"Query refined: '{query}' -> '{refined_query}'")
                
            return refined_query
            
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return query
