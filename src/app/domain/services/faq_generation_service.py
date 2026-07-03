"""FAQ AI 생성 서비스"""
import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta

from app.domain.constants import FAQ_MIN_CONFIDENCE, FAQ_MIN_FREQUENCY
from app.domain.ports.repository_port import FAQRepositoryPort
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.similarity_checker_port import SimilarityCheckerPort
from app.domain.ports.question_log_port import QuestionLogPort
from app.domain.ports.notification_port import NotificationPort
from app.infrastructure.persistence.models.faq_model import FaqStatus
from app.domain.utils.math_utils import cosine_similarity
from sqlalchemy.orm import Session
import numpy as np

logger = logging.getLogger(__name__)


class FAQGenerationService:
    """FAQ 자동 생성 서비스
    
    RAG 응답이 높은 신뢰도일 때 자동으로 FAQ 후보를 생성합니다.
    코사인 유사도 기반 중복 검사를 사용하여 의미적으로 유사한 FAQ를 정확히 감지합니다.
    
    임베딩 모델: Amazon Titan Embeddings v2 (amazon.titan-embed-text-v2:0)
    - FAQ 유사도 검사 시 사용
    - FAQ 검색 시 사용
    """
    
    def __init__(
        self,
        faq_repository: FAQRepositoryPort,
        llm_provider: LLMPort,
        similarity_checker: SimilarityCheckerPort,
        question_log_service: Optional[QuestionLogPort] = None,
        notification_service: Optional[NotificationPort] = None,
        db_session: Optional[Session] = None
    ):
        """FAQ 생성 서비스 초기화
        
        Args:
            faq_repository: FAQ Repository 포트
            llm_provider: LLM Provider 포트
            similarity_checker: 유사도 검사 포트
            question_log_service: 질문 로그 포트 (선택적)
            notification_service: 알림 서비스 포트 (선택적)
            db_session: 데이터베이스 세션 (선택적, 하위 호환성)
        """
        self.faq_repo = faq_repository
        self.llm_provider = llm_provider
        self.faq_similarity_checker = similarity_checker
        self.question_log_service = question_log_service
        self.notification_service = notification_service
        self.db_session = db_session  # 하위 호환성을 위해 유지
    
    async def generate_faq_candidate(
        self,
        user_question: str,
        rag_answer: str,
        rag_confidence: float,
        company_id: Optional[str] = None,
        min_confidence: float = 0.75,
        min_similarity_threshold: float = 0.7,
        min_frequency: int = 3,
        question_embedding: Optional[List[float]] = None
    ) -> Tuple[bool, Optional[int], str]:
        """FAQ 후보 생성 (신뢰도 + 빈도수 기반)
        
        Args:
            user_question: 사용자 질문
            rag_answer: RAG로 생성된 답변
            rag_confidence: RAG 신뢰도
            company_id: 회사 ID
            min_confidence: FAQ 생성 최소 신뢰도 (기본: 0.85)
            min_similarity_threshold: 유사 질문 체크 임계값 (기본: 0.7)
            min_frequency: 최소 빈도수 (의미 기반 클러스터의 총 빈도수, 기본: 3)
            question_embedding: 질문 임베딩 (제공되면 재사용, None이면 생성)
            
        Returns:
            (생성 여부, FAQ ID, 메시지)
        """
        # 1단계: 신뢰도 체크
        if rag_confidence < min_confidence:
            return False, None, f"신뢰도가 낮아 FAQ 생성하지 않음 (신뢰도: {rag_confidence:.2f} < {min_confidence})"
        
        # 2단계: 의미 기반 클러스터 빈도수 확인
        cluster_frequency = 0
        try:
            if self.question_log_service:
                cluster_frequency = self.question_log_service.get_cluster_frequency(
                    question=user_question,
                    question_embedding=question_embedding,
                    company_id=company_id or "default",
                    similarity_threshold=0.8,  # 클러스터링 유사도 임계값
                    days_back=30
                )
            else:
                # 하위 호환성: question_log_service가 없으면 스킵
                cluster_frequency = 0
            
            logger.info(
                f"🔍 [FAQ] 빈도수 체크: 클러스터 빈도수={cluster_frequency}, "
                f"최소 빈도수={min_frequency}, 신뢰도={rag_confidence:.2f}"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ [FAQ] 빈도수 확인 실패 (임베딩 생성 실패 등): {e}")
            # 임베딩 생성 실패 등으로 빈도수 확인이 불가능하면 FAQ 생성하지 않음
            cluster_frequency = 0
        
        # 빈도수 체크 (예외 발생 시에도 체크)
        if cluster_frequency < min_frequency:
            logger.warning(
                f"⚠️ [FAQ] 빈도수 부족으로 FAQ 생성 스킵: "
                f"클러스터 빈도수={cluster_frequency} < {min_frequency}, "
                f"신뢰도={rag_confidence:.2f}"
            )
            return False, None, (
                f"빈도수가 낮아 FAQ 생성하지 않음 "
                f"(클러스터 빈도수: {cluster_frequency} < {min_frequency}, "
                f"신뢰도: {rag_confidence:.2f})"
            )
        
        logger.info(
            f"✅ [FAQ] FAQ 생성 조건 통과: 신뢰도={rag_confidence:.2f}, "
            f"클러스터 빈도수={cluster_frequency} (임계값: {min_frequency})"
        )
        
        # 3단계: 유사한 FAQ가 이미 존재하는지 확인 (코사인 유사도 기반)
        # SimilarityCheckerPort의 find_similar_question을 사용하여 의미적으로 유사한 FAQ를 정확히 감지
        similar_faqs = self.faq_similarity_checker.find_similar_question(
            question=user_question,
            company_id=company_id,
            threshold=min_similarity_threshold,
            limit=5,
            question_embedding=question_embedding,  # 재사용
            session=self.db_session  # Fix: DetachedInstanceError 방지를 위해 세션 전달
        )
        
        if similar_faqs:
            # 유사한 FAQ가 있으면 생성하지 않음
            # similar_faqs는 dict 또는 객체일 수 있으므로 id 추출 방식 조정
            if isinstance(similar_faqs[0], dict):
                similar_questions = [f.get('id') for f in similar_faqs if f.get('id')]
            else:
                similar_questions = [f.id for f in similar_faqs]  # Java 백엔드와 일치: id 사용
            return False, None, f"유사한 FAQ가 이미 존재함 (FAQ IDs: {similar_questions}, 유사도 임계값: {min_similarity_threshold:.2f})"
        
        # LLM으로 질문-답변 쌍 최적화
        try:
            optimized_qa = self._optimize_qa_pair(user_question, rag_answer)
            if not optimized_qa:
                return False, None, "FAQ 최적화 실패"
            
            optimized_question, optimized_answer = optimized_qa
            
            # FAQ 후보 생성 (PENDING 상태) - Java 백엔드와 일치: company_id 파라미터 없음
            faq_dict = self.faq_repo.create_pending(
                question=optimized_question,
                answer=optimized_answer,
                embedding=question_embedding  # 임베딩도 함께 저장
            )
            
            faq_id = faq_dict.get('id')
            
            logger.info(
                f"FAQ 후보 생성 완료: ID={faq_id}, "
                f"question='{optimized_question[:50]}...', "
                f"confidence={rag_confidence:.2f}"
            )
            
            # 알림 발송 (비동기)
            # Fix: 알림 발송 전 트랜잭션 커밋 (백엔드에서 조회 가능하도록)
            if self.db_session:
                self.db_session.commit()
                
            if self.notification_service:
                try:
                    await self.notification_service.send_faq_pending_notification(faq_id)
                except Exception as e:
                    logger.warning(f"FAQ 알림 발송 실패 (생성은 성공): {e}")
            
            return True, faq_id, f"FAQ 후보 생성 완료 (ID: {faq_id}, 상태: PENDING)"
            
        except Exception as e:
            logger.error(f"FAQ 생성 오류: {e}", exc_info=True)
            return False, None, f"FAQ 생성 중 오류 발생: {str(e)}"
    
    def _optimize_qa_pair(
        self,
        user_question: str,
        rag_answer: str
    ) -> Optional[Tuple[str, str]]:
        """LLM으로 질문-답변 쌍 최적화
        
        사용자 질문을 더 명확하고 일반적인 형태로 변환하고,
        답변도 FAQ에 적합한 형태로 정리합니다.
        
        Args:
            user_question: 원본 질문
            rag_answer: 원본 답변
            
        Returns:
            (최적화된 질문, 최적화된 답변) 또는 None
        """
        try:
            llm = self.llm_provider.get_chat_llm(role="business")
            
            system_prompt = """당신은 FAQ 관리 전문가입니다.
사용자 질문과 답변을 받아서, FAQ에 적합한 형태로 최적화해주세요.

**질문 최적화 규칙:**
1. 구체적인 인명, 날짜, 숫자 등은 일반적인 표현으로 변경
2. "~하는 방법", "~절차", "~신청" 등 명확한 형태로 정리
3. 불필요한 감탄사나 불확실한 표현 제거
4. 간결하고 명확하게 작성

**답변 최적화 규칙:**
1. 핵심 정보만 포함
2. 단계별 설명이 있으면 명확하게 정리
3. 불필요한 출처 언급 제거 (FAQ에서는 출처 표시 불필요)
4. 간결하고 이해하기 쉽게 작성

**출력 형식:**
질문: [최적화된 질문]
답변: [최적화된 답변]"""
            
            user_prompt = f"""원본 질문: {user_question}

원본 답변:
{rag_answer}

위 질문과 답변을 FAQ에 적합한 형태로 최적화해주세요."""
            
            from langchain_core.messages import SystemMessage, HumanMessage
            
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content.strip()
            
            # 응답 파싱
            lines = content.split('\n')
            question = None
            answer = None
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('질문:'):
                    question = line.replace('질문:', '').strip()
                    current_section = 'question'
                elif line.startswith('답변:'):
                    answer = line.replace('답변:', '').strip()
                    current_section = 'answer'
                elif current_section == 'question' and question:
                    question += ' ' + line
                elif current_section == 'answer' and answer:
                    answer += ' ' + line
            
            if question and answer:
                return question.strip(), answer.strip()
            else:
                # 파싱 실패 시 원본 사용
                logger.warning(f"FAQ 최적화 파싱 실패, 원본 사용: {content[:100]}")
                return user_question, rag_answer
                
        except Exception as e:
            logger.error(f"FAQ 최적화 오류: {e}", exc_info=True)
            return None
    async def generate_faqs_from_frequency(
        self,
        company_id: str = "default",
        days_back: int = 30,
        min_frequency: int = FAQ_MIN_FREQUENCY,
        min_confidence: float = FAQ_MIN_CONFIDENCE,
        cluster_similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """빈도수 기반 FAQ 자동 생성 (의미 기반 클러스터링)
        
        Args:
            company_id: 회사 ID
            days_back: 분석 기간 (일)
            min_frequency: 최소 빈도수 (의미 기반 클러스터의 총 빈도수)
            min_confidence: RAG 최소 신뢰도 (기본: FAQ_MIN_CONFIDENCE)
            cluster_similarity_threshold: 클러스터링 유사도 임계값 (기본: 0.8)
            
        Returns:
            {
                "total_clusters": int,
                "faqs_created": int,
                "clusters_processed": int,
                "errors": List[str]
            }
        """
        try:
            from app.infrastructure.adapters.cache.question_log_service import QuestionLogService
            from app.infrastructure.adapters.cache import CacheService
            from app.infrastructure.adapters.rag import RAGService
            from app.infrastructure.persistence.session import db_session
            
            # 질문 로그 서비스 초기화
            cache_service = CacheService(use_redis=True)
            question_log_service = QuestionLogService(
                redis_client=cache_service.redis_client if cache_service.use_redis else None
            )
            
            # 30일 이내 질문 로그 조회 (성능 최적화: limit을 1000으로 제한)
            # O(n²) 알고리즘 특성상 10000개는 약 1억 회 계산이 필요하므로 1000개로 제한
            # 1000개면 약 100만 회 계산으로 충분히 처리 가능
            logs = question_log_service.get_all_logs(
                company_id=company_id,
                limit=1000,  # 성능 최적화: 10000 → 1000 (계산량 100배 감소)
                min_frequency=1,  # 클러스터링을 위해 모든 로그 조회
                days_back=days_back
            )
            
            if not logs:
                logger.info(f"FAQ 빈도 기반 생성: 질문 로그가 없습니다 (company_id={company_id})")
                return {
                    "total_clusters": 0,
                    "faqs_created": 0,
                    "clusters_processed": 0,
                    "errors": []
                }
            
            logger.info(f"FAQ 빈도 기반 생성 시작: 총 {len(logs)}개 질문 로그")
            
            # 임베딩이 있는 로그만 필터링
            logs_with_embedding = [
                log for log in logs 
                if log.get("embedding") is not None and isinstance(log.get("embedding"), list)
            ]
            
            if not logs_with_embedding:
                logger.warning("FAQ 빈도 기반 생성: 임베딩이 있는 로그가 없습니다. 임베딩 생성이 필요합니다.")
                return {
                    "total_clusters": 0,
                    "faqs_created": 0,
                    "clusters_processed": 0,
                    "errors": ["임베딩이 있는 로그가 없습니다"]
                }
            
            # 의미 기반 클러스터링
            clusters = self._cluster_questions_by_similarity(
                logs_with_embedding,
                similarity_threshold=cluster_similarity_threshold
            )
            
            logger.info(f"FAQ 빈도 기반 생성: {len(clusters)}개 클러스터 생성됨")
            
            # 빈도 합산 및 필터링
            cluster_stats = []
            for cluster_id, cluster_logs in clusters.items():
                total_frequency = sum(log["frequency"] for log in cluster_logs)
                if total_frequency >= min_frequency:
                    # 대표 질문 선택 (빈도가 가장 높은 것)
                    representative = max(cluster_logs, key=lambda x: x["frequency"])
                    cluster_stats.append({
                        "cluster_id": cluster_id,
                        "logs": cluster_logs,
                        "total_frequency": total_frequency,
                        "representative_question": representative["question"],
                        "representative_key": representative["key"]
                    })
            
            logger.info(f"FAQ 빈도 기반 생성: 빈도 {min_frequency}회 이상 클러스터 {len(cluster_stats)}개")
            
            # RAG 서비스 초기화
            rag_service = RAGService()
            
            faqs_created = 0
            errors = []
            
            # 각 클러스터에 대해 FAQ 생성
            for cluster_stat in cluster_stats:
                try:
                    question = cluster_stat["representative_question"]
                    cluster_logs = cluster_stat["logs"]
                    total_frequency = cluster_stat["total_frequency"]
                    
                    # RAG로 답변 생성
                    has_answer, rag_answer, sources, confidence, _, _ = rag_service.search(
                        query=question,
                        chat_history=None,
                        top_k=5
                    )
                    
                    if not has_answer or confidence < min_confidence:
                        logger.info(
                            f"FAQ 생성 스킵: 클러스터 {cluster_stat['cluster_id']}, "
                            f"신뢰도 부족 (confidence={confidence:.2f} < {min_confidence:.2f}), "
                            f"질문='{question[:50]}...', 빈도수={total_frequency}"
                        )
                        continue
                    
                    # 유사한 FAQ가 이미 존재하는지 확인
                    similar_faqs = self.faq_similarity_checker.find_similar_question(
                        question=question,
                        company_id=company_id,
                        threshold=0.7,
                        limit=5,
                        session=self.db_session
                    )
                    
                    if similar_faqs:
                        logger.info(
                            f"FAQ 생성 스킵: 클러스터 {cluster_stat['cluster_id']}, "
                            f"유사한 FAQ 이미 존재 (IDs: {[f.id for f in similar_faqs]}), "  # Java 백엔드와 일치: id 사용
                            f"질문='{question[:50]}...', 빈도수={total_frequency}"
                        )
                        # 클러스터 처리 완료 표시
                        question_log_service.update_cluster(
                            [log["key"] for log in cluster_logs],
                            cluster_stat["cluster_id"]
                        )
                        continue
                    
                    # FAQ 후보 생성
                    optimized_qa = self._optimize_qa_pair(question, rag_answer)
                    if not optimized_qa:
                        errors.append(f"FAQ 최적화 실패: {question[:50]}...")
                        continue
                    
                    optimized_question, optimized_answer = optimized_qa
                    
                    faq_dict = self.faq_repo.create_pending(
                        question=optimized_question,
                        answer=optimized_answer
                    )
                    
                    faq_id = faq_dict.get('id')
                    
                    # 클러스터 처리 완료 표시 (하위 호환성: question_log_service가 있으면)
                    if self.question_log_service and hasattr(self.question_log_service, 'update_cluster'):
                        try:
                            self.question_log_service.update_cluster(
                                [log["key"] for log in cluster_logs],
                                cluster_stat["cluster_id"]
                            )
                        except Exception as e:
                            logger.warning(f"클러스터 업데이트 실패: {e}")
                    
                    faqs_created += 1
                    logger.info(
                        f"FAQ 빈도 기반 생성 완료: ID={faq_id}, "
                        f"question='{optimized_question[:50]}...', "
                        f"frequency={total_frequency}, confidence={confidence:.2f}"
                    )
                    
                    # 알림 발송
                    if self.notification_service:
                        try:
                            await self.notification_service.send_faq_pending_notification(faq_id)
                        except Exception as e:
                            logger.warning(f"FAQ 알림 발송 실패 (생성은 성공): {e}")
                    
                except Exception as e:
                    error_msg = f"클러스터 {cluster_stat['cluster_id']} 처리 중 오류: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
                    continue
            
            return {
                "total_clusters": len(clusters),
                "faqs_created": faqs_created,
                "clusters_processed": len(cluster_stats),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"FAQ 빈도 기반 생성 중 오류: {e}", exc_info=True)
            return {
                "total_clusters": 0,
                "faqs_created": 0,
                "clusters_processed": 0,
                "errors": [f"전체 오류: {str(e)}"]
            }
    
    def _cluster_questions_by_similarity(
        self,
        logs: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> Dict[int, List[Dict[str, Any]]]:
        """의미 기반 클러스터링 (코사인 유사도 기반, 배치 처리 최적화)
        
        임베딩 벡터 간 코사인 유사도를 계산하여 유사한 질문들을 클러스터로 묶습니다.
        numpy 벡터화 연산을 사용하여 성능을 최적화했습니다.
        
        Args:
            logs: 질문 로그 리스트 (embedding 필수)
            similarity_threshold: 클러스터링 유사도 임계값 (기본: 0.8)
            
        Returns:
            {cluster_id: [log1, log2, ...]} 딕셔너리
        """
        if not logs:
            return {}
        
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        cluster_id_counter = 0
        assigned = set()  # 이미 클러스터에 할당된 로그 인덱스
        
        # 임베딩 벡터를 numpy 배열로 변환
        embeddings = []
        valid_indices = []
        for i, log in enumerate(logs):
            embedding = log.get("embedding")
            if embedding and isinstance(embedding, list):
                embeddings.append(np.array(embedding, dtype=np.float32))  # float32로 메모리 최적화
                valid_indices.append(i)
        
        if not embeddings:
            return {}
        
        # 임베딩 벡터를 행렬로 변환 (배치 처리용)
        embedding_matrix = np.array(embeddings)
        embedding_dim = embedding_matrix.shape[1]
        
        # 벡터 정규화 (L2 norm) - 배치 코사인 유사도 계산을 위해
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 영벡터 방지
        normalized_embeddings = embedding_matrix / norms
        
        # 각 로그에 대해 클러스터 할당 (배치 처리 최적화)
        for idx, log_idx in enumerate(valid_indices):
            if log_idx in assigned:
                continue
            
            log = logs[log_idx]
            embedding = normalized_embeddings[idx]
            
            # 새 클러스터 생성
            cluster_id = cluster_id_counter
            cluster_id_counter += 1
            clusters[cluster_id] = [log]
            assigned.add(log_idx)
            
            # 배치 코사인 유사도 계산 (numpy 벡터화 연산)
            # 아직 할당되지 않은 임베딩들만 선택
            unassigned_indices = [
                i for i in range(len(valid_indices))
                if valid_indices[i] not in assigned and valid_indices[i] != log_idx
            ]
            
            if not unassigned_indices:
                continue
            
            # 배치 코사인 유사도 계산: dot product (정규화된 벡터이므로)
            unassigned_embeddings = normalized_embeddings[unassigned_indices]
            similarities = np.dot(unassigned_embeddings, embedding)
            
            # 임계값 이상인 인덱스 찾기
            similar_mask = similarities >= similarity_threshold
            similar_relative_indices = np.where(similar_mask)[0]
            
            # 유사한 질문들을 같은 클러스터에 추가
            for rel_idx in similar_relative_indices:
                abs_idx = unassigned_indices[rel_idx]
                other_log_idx = valid_indices[abs_idx]
                if other_log_idx not in assigned:
                    clusters[cluster_id].append(logs[other_log_idx])
                    assigned.add(other_log_idx)
        
        logger.info(
            f"클러스터링 완료: {len(clusters)}개 클러스터 생성 "
            f"(임계값: {similarity_threshold}, 처리된 로그: {len(valid_indices)}개)"
        )
        return clusters

