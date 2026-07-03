"""AWS Bedrock Rerank 기반 Reranker

AWS Bedrock의 Rerank API를 사용하여 검색 결과를 재정렬합니다.
"""
from __future__ import annotations

import logging
import json
import time
import math
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class BedrockReranker:
    """AWS Bedrock Rerank 기반 Reranker
    
    AWS Bedrock의 Rerank API를 사용하여 검색 결과를 재정렬하고
    더 정확한 신뢰도 점수를 계산합니다.
    
    지원 모델:
    - amazon.rerank-v1:0 (Amazon Rerank 1.0)
    - cohere.rerank-v3-5:0 (Cohere Rerank 3.5)
    
    특징:
    - 서버리스 실행 (로컬 리소스 불필요)
    - GPU 최적화 (빠른 처리 속도)
    - 다국어 지원
    
    Example:
        ```python
        reranker = BedrockReranker(
            model_id="amazon.rerank-v1:0",
            region="ap-northeast-1"
        )
        reranked_docs = reranker.rerank(
            query="회사 소개",
            documents=[doc1, doc2, doc3],
            top_k=5
        )
        ```
    """
    
    def __init__(
        self,
        model_id: str = "amazon.rerank-v1:0",
        region: Optional[str] = None,  # Security Fix: 하드코딩 제거, 필수 파라미터로 변경
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        top_n: int = 10
    ):
        """BedrockReranker 초기화
        
        Args:
            model_id: Bedrock Rerank 모델 ID
                - 짧은 형식: "amazon.rerank-v1:0" (기본값, 자동으로 현재 region에 맞는 ARN으로 변환됨)
                - ARN 형식: "arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.rerank-v1:0" (현재 프로젝트 region)
                - 공식 예제 형식: "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0" (자동으로 현재 region으로 변환됨)
                - "cohere.rerank-v3-5:0": Cohere Rerank 3.5
            region: AWS 리전 (기본값: "ap-northeast-1")
            aws_access_key_id: AWS Access Key ID (선택적, 환경변수 사용 가능)
            aws_secret_access_key: AWS Secret Access Key (선택적, 환경변수 사용 가능)
            top_n: 최대 재정렬할 문서 수 (기본값: 10)
        
        Note:
            - 짧은 형식을 사용하면 자동으로 현재 프로젝트의 region(ap-northeast-1)에 맞는 ARN 형식으로 변환됩니다.
            - 공식 예제의 ARN 형식(us-west-2)을 사용해도 자동으로 현재 프로젝트의 region으로 변환됩니다.
            - 이렇게 하면 공식 예제와의 호환성을 유지하면서 현재 프로젝트의 region에 맞게 작동합니다.
        """
        # Security Fix: region이 필수인 경우 검증
        if not region:
            raise ValueError("AWS 리전이 설정되지 않았습니다. region 파라미터를 필수로 제공하세요.")
        
        self.model_id = model_id
        self.region = region
        self.top_n = top_n
        self._client = None
        self._initialized = False
        
        # boto3 세션 생성
        try:
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError
            
            if aws_access_key_id and aws_secret_access_key:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region
                )
                logger.info(f"BedrockReranker: 자격증명 사용 (access_key_id={aws_access_key_id[:10]}...)")
            else:
                session = boto3.Session(region_name=region)
                logger.info("BedrockReranker: 환경변수/프로파일 자격증명 사용")
            
            self._client = session.client("bedrock-runtime", region_name=region)
            self._initialized = True
            logger.info(f"BedrockReranker 초기화 완료: model_id={model_id}, region={region}")
        except ImportError:
            raise ImportError(
                "boto3 패키지가 설치되지 않았습니다.\n"
                "설치: pip install boto3"
            )
        except Exception as e:
            logger.error(f"BedrockReranker 초기화 실패: {e}")
            raise RuntimeError(f"BedrockReranker 초기화 실패: {e}") from e
    
    def _ensure_document_format(self, documents: List[Any]) -> List[Dict[str, str]]:
        """문서 입력을 안전한 payload 형식으로 변환
        
        입력은 세 가지 허용:
         - dict with 'text' key
         - langchain Document-like object with .page_content
         - raw string
        
        Returns:
            [{'text': str}, ...]  # Bedrock Rerank API는 metadata를 지원하지 않음
        """
        out = []
        for i, d in enumerate(documents):
            if isinstance(d, dict) and "text" in d:
                text = d["text"]
            elif hasattr(d, "page_content"):
                text = getattr(d, "page_content")
            else:
                # fallback to string
                text = str(d)
            
            if text is None:
                text = ""
            text = text.strip()
            
            # 길이 제한 (Reranker 성능을 위해)
            max_len = 5000
            if len(text) > max_len:
                text = text[:max_len] + "..."
            
            # Bedrock Rerank API는 {"text": "..."} 형식만 지원 (metadata 불가)
            out.append({"text": text})
        
        return out
    
    def _to_arn_if_needed(self, model_id: str) -> str:
        """모델 ID를 ARN 형식으로 변환 (필요시)
        
        Args:
            model_id: 모델 ID (짧은 형식 또는 ARN 형식)
        
        Returns:
            ARN 형식의 모델 ID
        """
        if model_id.startswith("arn:aws:bedrock"):
            # ARN 형식이면 region 확인
            arn_parts = model_id.split(":")
            if len(arn_parts) >= 4:
                arn_region = arn_parts[3]
                if arn_region != self.region:
                    logger.warning(
                        f"BedrockReranker: ARN의 region({arn_region})과 설정된 region({self.region})이 다릅니다. "
                        f"현재 프로젝트의 region({self.region})에 맞게 ARN을 재생성합니다."
                    )
                    # 모델 ID 추출 (ARN에서 마지막 부분)
                    model_id_from_arn = model_id.split("/")[-1] if "/" in model_id else model_id
                    # 현재 region에 맞게 ARN 재생성
                    return f"arn:aws:bedrock:{self.region}::foundation-model/{model_id_from_arn}"
            return model_id
        
        # 짧은 형식을 현재 프로젝트의 region에 맞는 ARN 형식으로 변환
        return f"arn:aws:bedrock:{self.region}::foundation-model/{model_id}"
    
    def call_reranker(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Bedrock invoke wrapper
        
        이 메서드는 boto3 bedrock-runtime client의 반환 구조에 맞춰 StreamingBody를 읽어
        JSON으로 디코딩하여 반환합니다.
        
        Args:
            request_body: Bedrock Rerank API 요청 본문
        
        Returns:
            파싱된 JSON 응답
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("BedrockReranker가 초기화되지 않았습니다.")
        
        model_arn = self._to_arn_if_needed(self.model_id)
        
        try:
            logger.debug(
                f"BedrockReranker API 호출: model={model_arn}, documents={len(request_body.get('documents', []))}"
            )
            response = self._client.invoke_model(
                modelId=model_arn,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
        except Exception as e:
            logger.exception(f"BedrockReranker invoke_model 실패: {e}")
            raise
        
        # StreamingBody 읽기
        if "body" not in response:
            logger.error(f"BedrockReranker 응답에 'body' 키가 없습니다: {list(response.keys())}")
            raise RuntimeError("Invalid bedrock response: missing body")
        
        try:
            body_stream = response["body"]
            # boto3 StreamingBody에는 read()가 있음
            raw = body_stream.read()
            
            # bytes 처리
            if isinstance(raw, bytes):
                raw_text = raw.decode("utf-8")
            else:
                raw_text = str(raw)
            
            parsed = json.loads(raw_text)
            return parsed
        except Exception as e:
            logger.exception(f"BedrockReranker 응답 읽기/파싱 실패: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        scores_only: bool = False
    ) -> List[Tuple[Document, float]]:
        """검색 결과 재정렬 및 점수 재계산

        Args:
            query: 검색 쿼리
            documents: 재정렬할 문서 리스트
            top_k: 반환할 상위 문서 수 (None이면 모두 반환)
            scores_only: 점수만 반환할지 여부 (현재는 미사용, 향후 확장용)

        Returns:
            (Document, score) 튜플 리스트 (점수 내림차순 정렬)
            - Document: 재정렬된 문서
            - score: Reranker 점수 (float, 높을수록 관련성 높음)
              * Cohere Rerank 3.5: 0~1 사이의 정규화된 점수
              * Amazon Rerank 1.0: 로그잇 기반 점수(범위 고정 X, 상대 비교용)
        """
        if not documents:
            return []

        if not self._initialized or not self._client:
            raise RuntimeError("BedrockReranker가 초기화되지 않았습니다.")

        start_time = time.time()
        is_cohere = "cohere" in self.model_id.lower()

        try:
            # 문서 형식 변환 (dict/Document/string 모두 지원, metadata 보존)
            docs_payload = self._ensure_document_format(documents)

            if not docs_payload:
                logger.warning("BedrockReranker: 처리할 문서가 없습니다")
                return []

            # top_n은 최대 재정렬할 문서 수 (Bedrock API 제한)
            # Bedrock Rerank API는 최대 100개 문서까지 처리 가능
            actual_top_n = min(self.top_n, len(docs_payload), 100)
            payload_docs = docs_payload[:actual_top_n]

            # 1) 요청 바디 구성
            # - Amazon Rerank: {"query": "...", "documents": [{"text": "..."}, ...]}
            # - Cohere Rerank v3-5: {"api_version": 2, "query": "...", "documents": ["...", ...]}
            if is_cohere:
                request_body = {
                    "api_version": 2,
                    "query": query,
                    "documents": [doc["text"] for doc in payload_docs],  # 문자열 배열
                }
            else:
                request_body = {
                    "query": query,
                    "documents": payload_docs,  # [{"text": "..."}, ...]
                }

            # 2) 요청 바디 검증
            docs_field = request_body.get("documents")
            if not isinstance(docs_field, list):
                raise ValueError("documents는 리스트여야 합니다")
            if len(docs_field) == 0:
                raise ValueError("documents가 비어있습니다")

            if is_cohere:
                # Cohere Rerank: documents는 문자열 배열이어야 함
                for i, doc in enumerate(docs_field):
                    if not isinstance(doc, str):
                        raise ValueError(
                            f"Cohere Rerank: documents[{i}]는 문자열이어야 합니다 "
                            f"(현재 타입: {type(doc)})"
                        )
            else:
                # Amazon Rerank: documents는 {"text": "..."} 딕셔너리 배열
                for i, doc in enumerate(docs_field):
                    if not isinstance(doc, dict):
                        raise ValueError(
                            f"Amazon Rerank: documents[{i}]는 딕셔너리여야 합니다 "
                            f"(현재 타입: {type(doc)})"
                        )
                    if "text" not in doc:
                        raise ValueError(
                            f"Amazon Rerank: documents[{i}]에 'text' 키가 없습니다"
                        )
                    if not isinstance(doc["text"], str):
                        raise ValueError(
                            f"Amazon Rerank: documents[{i}]['text']는 문자열이어야 합니다"
                        )

            # 3) 디버깅 로그
            model_arn = self._to_arn_if_needed(self.model_id)
            logger.info(
                f"BedrockReranker 요청 준비: model_id='{model_arn}', "
                f"query='{query[:80]}...', documents={len(payload_docs)}, "
                f"format={'Cohere (문자열 배열)' if is_cohere else 'Amazon (객체 배열)'}"
            )
            if payload_docs:
                if is_cohere:
                    first_doc_text = docs_field[0] if docs_field else ""
                else:
                    first_doc_text = payload_docs[0].get("text", "")
                logger.debug(
                    f"BedrockReranker 첫 번째 문서 샘플: {first_doc_text[:200]}..."
                )

            # 4) Bedrock Rerank API 호출
            response_body = self.call_reranker(request_body)

            # 5) 응답 구조 로깅
            logger.info(
                f"BedrockReranker 전체 응답 키: {list(response_body.keys())}"
            )

            try:
                response_body_str = json.dumps(
                    response_body, indent=2, ensure_ascii=False
                )
                if len(response_body_str) > 2000:
                    logger.debug(
                        f"BedrockReranker 전체 응답 (첫 2000자): "
                        f"{response_body_str[:2000]}..."
                    )
                else:
                    logger.debug(
                        f"BedrockReranker 전체 응답: {response_body_str}"
                    )
            except Exception:
                # logging 실패는 무시
                pass

            # 6) 결과 필드 추출
            raw_results = response_body.get("results")
            if raw_results is None:
                logger.warning(
                    "BedrockReranker 응답에 'results' 필드가 없습니다. "
                    f"사용 가능한 키: {list(response_body.keys())}. "
                    "items/candidates 필드를 순차적으로 시도합니다."
                )
                raw_results = (
                    response_body.get("items")
                    or response_body.get("candidates")
                    or []
                )

            if not raw_results:
                logger.error(
                    "BedrockReranker 응답에 results/items/candidates가 모두 없습니다. "
                    f"전체 응답 키: {list(response_body.keys())}"
                )
                # 폴백: 정렬 없이 원본 순서대로 반환 (점수 0.0)
                logger.warning("Reranker 응답이 비어있습니다. 원본 순서대로 반환합니다.")
                base = list(zip(documents, [0.0] * len(documents)))
                return base[:top_k] if top_k else base

            logger.info(f"BedrockReranker results 수: {len(raw_results)}")

            # 7) 결과 파싱
            parsed_results: List[Dict[str, Any]] = []
            for i, item in enumerate(raw_results):
                if i == 0:
                    logger.info(
                        "BedrockReranker 첫 번째 결과: "
                        f"{json.dumps(item, indent=2, ensure_ascii=False)}"
                    )

                # 인덱스 추출
                index = item.get("index", i)
                if not isinstance(index, int):
                    logger.warning(
                        f"BedrockReranker 결과 항목 index 타입이 int가 아닙니다: {type(index)}"
                    )
                    try:
                        index = int(index)
                    except Exception:
                        # 인덱스가 이상하면 이 항목은 건너뜀
                        continue

                # 점수 추출 (키 후보: relevance_score, relevanceScore, score)
                score = None
                for key in ("relevance_score", "relevanceScore", "score"):
                    if key in item:
                        score = item[key]
                        break

                if score is None:
                    logger.warning(
                        f"BedrockReranker 결과 항목에 점수 필드가 없습니다: keys={list(item.keys())}"
                    )
                    continue

                try:
                    score = float(score)
                except Exception:
                    logger.warning(
                        f"BedrockReranker 점수 값을 float로 변환할 수 없습니다: {score}"
                    )
                    continue

                if math.isnan(score) or math.isinf(score):
                    logger.warning(
                        f"BedrockReranker 점수가 NaN/Inf 입니다. 0.0으로 대체합니다: {score}"
                    )
                    score = 0.0
                
                # Amazon Rerank (non-Cohere) 모델의 경우 Logits를 Sigmoid로 정규화
                if not is_cohere:
                    # Sigmoid 변환: 1 / (1 + exp(-x))
                    # Amazon Rerank는 Logits를 반환하므로 0~1 범위로 변환 필요
                    try:
                        # overflow 방지를 위한 클리핑 (-100 ~ 100)
                        clipped_score = max(-100.0, min(100.0, score))
                        normalized_score = 1.0 / (1.0 + math.exp(-clipped_score))
                        
                        # 디버깅 로깅 (첫 번째 항목만)
                        if i == 0:
                            logger.info(f"Amazon Rerank 정규화: raw={score:.4f} -> normalized={normalized_score:.4f}")
                            
                        score = normalized_score
                    except Exception as e:
                        logger.warning(f"Sigmoid 변환 실패: {e}, 원본 점수 사용: {score}")

                parsed_results.append({"index": index, "score": score})

            if not parsed_results:
                logger.warning("BedrockReranker: 파싱된 결과가 없습니다. 원본 순서 사용.")
                base = list(zip(documents, [0.0] * len(documents)))
                return base[:top_k] if top_k else base

            # score 기준 내림차순 정렬 (모델이 정렬해주더라도 한 번 더 보정)
            parsed_results.sort(key=lambda x: x["score"], reverse=True)

            # 8) 원본 documents의 인덱스를 올바르게 매핑
            reranked_docs: List[Tuple[Document, float]] = []
            for r in parsed_results:
                idx = r["index"]
                score = r["score"]

                if 0 <= idx < len(documents):
                    original_obj = documents[idx]
                    reranked_docs.append((original_obj, score))
                else:
                    logger.warning(
                        "BedrockReranker: 유효하지 않은 인덱스 %s "
                        "(문서 수: %s, payload_docs 수: %s)",
                        idx,
                        len(documents),
                        len(payload_docs),
                    )

            if not reranked_docs:
                logger.warning("BedrockReranker: 재정렬된 문서가 없습니다")
                raise ValueError("BedrockReranker: 재정렬된 문서가 없습니다")

            # top_k만큼 잘라서 반환
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]

            elapsed_time = time.time() - start_time

            # 점수 통계
            final_scores = [s for _, s in reranked_docs]
            max_final_score = max(final_scores)
            min_final_score = min(final_scores)

            logger.info(
                f"BedrockReranker 재정렬 완료: 쿼리='{query[:50]}...', "
                f"입력={len(payload_docs)}개, 출력={len(reranked_docs)}개, "
                f"최고 점수={max_final_score:.6f}, 최저 점수={min_final_score:.6f}, "
                f"소요 시간={elapsed_time:.3f}초"
            )

            if len(reranked_docs) >= 3:
                top_3_scores = [f"{s:.6f}" for _, s in reranked_docs[:3]]
                logger.info(f"BedrockReranker 상위 3개 점수: {', '.join(top_3_scores)}")

            return reranked_docs

        except Exception as e:
            logger.error(f"BedrockReranker 점수 계산 실패: {e}", exc_info=True)
            # 폴백: 원본 순서 유지 + 점수 0.0
            base = list(zip(documents, [0.0] * len(documents)))
            return base[:top_k] if top_k else base
    
    def rerank_with_original_scores(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """원본 점수와 함께 재정렬 (비교용)
        
        Args:
            query: 검색 쿼리
            documents_with_scores: (Document, original_score) 튜플 리스트
            top_k: 반환할 상위 문서 수 (None이면 모두 반환)
            
        Returns:
            (Document, original_score, reranker_score) 튜플 리스트
            - Document: 재정렬된 문서
            - original_score: 원본 점수 (Bedrock KB 또는 하이브리드)
            - reranker_score: Reranker 점수
        """
        documents = [doc for doc, _ in documents_with_scores]
        original_scores = [score for _, score in documents_with_scores]
        
        # Reranker로 재정렬
        reranked = self.rerank(query, documents, top_k=top_k)
        
        # 원본 점수 매핑 (Document는 hashable하지 않으므로 인덱스 기반 매핑 사용)
        # Document의 page_content를 키로 사용 (내용이 같으면 같은 문서로 간주)
        doc_to_original_score = {}
        for doc, score in documents_with_scores:
            # page_content를 키로 사용 (내용 기반 매핑)
            doc_key = doc.page_content[:100] if doc.page_content else ""  # 처음 100자만 사용
            if doc_key not in doc_to_original_score:
                doc_to_original_score[doc_key] = score
        
        # 결과 생성
        results = []
        for doc, reranker_score in reranked:
            # 재정렬된 문서의 page_content로 원본 점수 찾기
            doc_key = doc.page_content[:100] if doc.page_content else ""
            original_score = doc_to_original_score.get(doc_key, 0.0)
            results.append((doc, original_score, reranker_score))
        
        return results

    def compute_combined_confidence(
        self, 
        kb_best_score: float, 
        reranker_best_score: float, 
        alpha: float = 0.6
    ) -> float:
        """단순 가중치 결합: final = alpha*kb + (1-alpha)*reranker
        
        Args:
            kb_best_score: embedding 기반 KB 검색의 최고 점수 (0~1 가정)
            reranker_best_score: reranker에서 얻은 raw score (범위가 모델마다 다름)
            alpha: KB 점수 가중치 (기본값: 0.6)
        
        Returns:
            결합된 신뢰도 점수
        
        Note:
            - reranker score가 0~1을 벗어날 수 있으므로, 사용자는 reranker score 스케일을 알고 있어야 합니다.
            - 본 헬퍼는 기본적으로 reranker score를 그대로 사용하되, 필요시 사용자 측에서 재스케일링을 권장합니다.
        """
        try:
            kb = float(kb_best_score)
        except Exception:
            kb = 0.0
        
        try:
            rr = float(reranker_best_score)
        except Exception:
            rr = 0.0
        
        # KB 점수는 안전하게 클램프 (0~1 범위)
        kb = max(0.0, min(1.0, kb))
        
        # reranker는 범위가 다양하므로 사용자 정의 스케일링 권장
        # 여기서는 단순 클램프를 수행하지 않음(원본 스케일 유지)
        combined = alpha * kb + (1.0 - alpha) * rr
        
        return combined

    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        scores_only: bool = False
    ) -> List[Tuple[Document, float]]:
        """비동기 검색 결과 재정렬"""
        import asyncio
        # I/O 바운드 작업(API 호출)이지만 boto3가 동기식이므로 스레드 풀에서 실행
        return await asyncio.to_thread(
            self.rerank, query, documents, top_k, scores_only
        )

    async def arerank_with_original_scores(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """비동기 원본 점수와 함께 재정렬"""
        import asyncio
        return await asyncio.to_thread(
            self.rerank_with_original_scores, query, documents_with_scores, top_k
        )

        
        return combined

