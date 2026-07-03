"""FAQ 유사도 검사기 (PostgreSQL 사용) - FAQ 생성 시 중복 검사용"""
from typing import Optional, List
import logging
import numpy as np
from sqlalchemy.orm import Session

from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
from app.infrastructure.persistence.session import db_session, SessionLocal
from app.infrastructure.persistence.models.faq_model import FaqStatus
from app.infrastructure.adapters.cache import HybridCache
from app.infrastructure.utils.utils_math import cosine_similarity
from app.infrastructure.config.config_loader import get_config_int

logger = logging.getLogger(__name__)


class FAQSimilarityChecker:
    """FAQ 유사도 검사기 (PostgreSQL 기반)
    
    FAQ 생성 시 유사 질문 중복 검사에만 사용됩니다.
    코사인 유사도 기반으로 의미적으로 유사한 FAQ를 찾습니다.
    
    일반 FAQ 검색은 FAQServiceRedis를 사용합니다.
    """

    def __init__(self, session: Optional[Session] = None):
        """FAQ 서비스 초기화
        
        Args:
            session: SQLAlchemy 세션 (None이면 자동 생성)
        """
        self._external_session = session
        
        # 임베딩 모델 초기화 (Cosine Similarity용)
        # Amazon Titan Embeddings v2 사용 (amazon.titan-embed-text-v2:0)
        try:
            from app.infrastructure.adapters.llm import LLMProvider
            self.embeddings = LLMProvider().get_embeddings()
            # FAQ 임베딩 캐시 (LRU + TTL 하이브리드)
            max_cache_size = get_config_int("FAQ_EMBEDDING_CACHE_SIZE", 1000, section="faq")
            cache_ttl = get_config_int("FAQ_EMBEDDING_CACHE_TTL", 3600, section="faq")  # 1시간
            self._faq_embeddings_cache = HybridCache(
                max_size=max_cache_size,
                ttl_seconds=cache_ttl
            )
            logger.info(f"FAQ 임베딩 초기화 완료: model=amazon.titan-embed-text-v2:0, cache_size={max_cache_size}, ttl={cache_ttl}s")
        except Exception as e:
            logger.warning(f"임베딩 모델 초기화 실패: {e}. 키워드 매칭으로 폴백합니다.")
            self.embeddings = None
            self._faq_embeddings_cache = None

    def find_similar_question(
        self,
        question: str,
        company_id: Optional[str] = None,
        threshold: float = 0.8,
        limit: int = 5,
        session: Optional[Session] = None,
        question_embedding: Optional[np.ndarray] = None
    ) -> List:
        """유사한 질문이 이미 존재하는지 확인 (코사인 유사도 기반)"""
        if session:
            return self._find_similar_question_with_session(
                session, question, company_id, threshold, limit, question_embedding
            )
        else:
            with db_session() as db:
                return self._find_similar_question_with_session(
                    db, question, company_id, threshold, limit, question_embedding
                )

    def _find_similar_question_with_session(
        self,
        session: Session,
        question: str,
        company_id: Optional[str],
        threshold: float,
        limit: int,
        question_embedding: Optional[np.ndarray]
    ) -> List:
        """세션이 보장된 상태에서 유사 질문 검색 수행"""
        try:
            # 임베딩 모델이 없으면 키워드 매칭으로 폴백
            if not self.embeddings or self._faq_embeddings_cache is None:
                return self._find_similar_question_keyword_fallback(
                    question, company_id, threshold, limit, session
                )
            
            # 1. 사용자 질문 임베딩 (제공되면 재사용, 없으면 생성)
            if question_embedding is None:
                question_embedding = np.array(self.embeddings.embed_query(question))
            else:
                question_embedding = np.array(question_embedding)  # numpy 배열로 변환
            
            # 2. 키워드로 후보 FAQ 필터링 (성능 최적화)
            import re
            particles = ['에', '에서', '에게', '을', '를', '이', '가', '은', '는', '와', '과', '도', '로', '으로', '의', 
                        '에 대해', '에 관한', '시', '어떻게', '무엇인가요', '하나요', '인가요', '는가요', '인지', '인지요']
            cleaned_message = question.lower()
            for particle in particles:
                cleaned_message = cleaned_message.replace(particle, ' ')
            keywords = [word.strip() for word in cleaned_message.split() if len(word.strip()) > 1]
            
            logger.info(f"[SimilarityCheck] Question: '{question}', Keywords: {keywords}")
            
            if not keywords:
                logger.info("[SimilarityCheck] No keywords extracted.")
                return []
            
            # Repository 사용
            # Fix: 중복 생성을 방지하기 위해 PENDING 상태의 FAQ도 함께 검사
            target_statuses = [FaqStatus.ACTIVE, FaqStatus.PENDING]
            
            repo = FaqRepository(session)
            faqs = repo.find_by_keywords(
                keywords=keywords[:5] if keywords else [],
                status=target_statuses,
                limit=50  # 더 많은 후보를 가져와서 코사인 유사도로 정확히 필터링
            )
            
            logger.info(f"[SimilarityCheck] Found {len(faqs)} candidates from DB using keywords.")
            
            if not faqs:
                return []
            
            # 3. 배치 임베딩 생성 및 코사인 유사도 계산
            faq_data_list = []
            faq_metadata = []  # (faq, cache_key, embedding) 튜플
            
            # Redis 클라이언트 가져오기 (사전 생성된 임베딩 사용)
            redis_client = None
            try:
                import redis
                from app.infrastructure.config.config_loader import get_config, get_config_int
                from redis.connection import ConnectionPool
                
                redis_pool = ConnectionPool(
                    host=get_config("host", "localhost", section="redis"),
                    port=get_config_int("port", 6379, section="redis"),
                    db=get_config_int("db", 0, section="redis"),
                    decode_responses=True
                )
                redis_client = redis.Redis(connection_pool=redis_pool)
            except Exception as e:
                logger.warning(f"Redis 연결 실패 (사전 생성 임베딩 사용 불가): {e}")
            
            for faq in faqs:
                faq_text = f"{faq.question} {faq.answer}"
                cache_key = f"{faq.id}_{faq.question}_{faq.answer}"  # Java 백엔드와 일치: id 사용
                
                # FAQ 임베딩 가져오기 (우선순위: Redis 사전 생성 > 인메모리 캐시 > 스킵)
                faq_embedding = None
                
                # Redis에서 사전 생성된 임베딩 확인
                if redis_client and company_id:
                    try:
                        embedding_key = f"faq_embedding:{company_id}:{faq.question}"
                        redis_embedding_json = redis_client.get(embedding_key)
                        if redis_embedding_json:
                            import json
                            redis_embedding = json.loads(redis_embedding_json)
                            faq_embedding = np.array(redis_embedding)
                            # 인메모리 캐시에도 저장
                            self._faq_embeddings_cache.set(cache_key, faq_embedding)
                    except Exception as e:
                        logger.debug(f"Redis 임베딩 가져오기 실패: {e}")
                
                # 인메모리 캐시 확인
                if faq_embedding is None:
                    cached_embedding = self._faq_embeddings_cache.get(cache_key)
                    if cached_embedding is not None:
                        faq_embedding = cached_embedding
                
                # Fix: 임베딩이 없으면 실시간 생성 (PENDING 상태 등 신규 FAQ를 위해 필수)
                if faq_embedding is None and self.embeddings:
                    try:
                        # logger.debug(f"FAQ 임베딩 실시간 생성: id={faq.id}")
                        faq_embedding = np.array(self.embeddings.embed_query(faq_text))
                        # 캐시에 저장
                        self._faq_embeddings_cache.set(cache_key, faq_embedding)
                    except Exception as e:
                        logger.warning(f"FAQ 임베딩 생성 실패 (스킵): id={faq.id}, error={e}")
                        continue

                if faq_embedding is not None:
                    faq_metadata.append((faq, cache_key, faq_embedding))
            
            # 4. 코사인 유사도 계산 및 임계값 필터링
            similar_faqs = []
            
            for faq, cache_key, faq_embedding in faq_metadata:
                if faq_embedding is None:
                    continue
                
                try:
                    score = cosine_similarity(question_embedding, faq_embedding)
                    
                    # 임계값 이상인 FAQ만 추가
                    if score >= threshold:
                        similar_faqs.append((faq, score))
                except Exception as e:
                    logger.warning(f"코사인 유사도 계산 실패: {e}")
                    continue
            
            # 5. 유사도 내림차순 정렬 및 limit 적용
            similar_faqs.sort(key=lambda x: x[1], reverse=True)
            result = [faq for faq, score in similar_faqs[:limit]]
            
            if result:
                logger.info(
                    f"유사 FAQ 발견: question='{question[:50]}...', "
                    f"count={len(result)}, threshold={threshold:.2f}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"유사 질문 검색 오류 (코사인 유사도): {e}", exc_info=True)
            # 오류 시 키워드 매칭으로 폴백
            return self._find_similar_question_keyword_fallback(
                question, company_id, threshold, limit, session
            )
    
    def _find_similar_question_keyword_fallback(
        self,
        question: str,
        company_id: Optional[str],
        threshold: float,
        limit: int,
        session: Session
    ) -> List:
        """유사 질문 검색 (키워드 매칭 폴백)
        
        임베딩 모델이 없거나 오류 발생 시 사용하는 폴백 메서드입니다.
        단순 키워드 매칭을 사용하므로 정확도가 낮을 수 있습니다.
        """
        logger.info(f"[유사 질문 검색] 키워드 매칭 폴백 사용")
        try:
            import re
            particles = ['에', '에서', '에게', '을', '를', '이', '가', '은', '는', '와', '과', '도', '로', '으로', '의', 
                        '에 대해', '에 관한', '시', '어떻게', '무엇인가요', '하나요', '인가요', '는가요', '인지', '인지요']
            cleaned_message = question.lower()
            for particle in particles:
                cleaned_message = cleaned_message.replace(particle, ' ')
            keywords = [word.strip() for word in cleaned_message.split() if len(word.strip()) > 1]
            
            if not keywords:
                return []
            
            # Fix: 중복 생성을 방지하기 위해 PENDING 상태의 FAQ도 함께 검사
            target_statuses = [FaqStatus.ACTIVE, FaqStatus.PENDING]
            
            repo = FaqRepository(session)
            faqs = repo.find_by_keywords(
                keywords=keywords,
                status=target_statuses,
                limit=limit * 2  # 더 많이 가져와서 점수 계산
            )
            
            if not faqs:
                return []
            
            # 키워드 매칭 점수 계산
            scored_faqs = []
            for faq in faqs:
                question_keywords = [kw for kw in keywords if kw in faq.question.lower()]
                answer_keywords = [kw for kw in keywords if kw in faq.answer.lower()]
                total_keywords = len(keywords)
                matched_keywords = len(set(question_keywords + answer_keywords))
                score = matched_keywords / total_keywords if total_keywords > 0 else 0.0
                
                if score >= threshold:
                    scored_faqs.append((faq, score))
            
            # 점수 내림차순 정렬 및 limit 적용
            scored_faqs.sort(key=lambda x: x[1], reverse=True)
            result = [faq for faq, score in scored_faqs[:limit]]
            
            return result
            
        except Exception as e:
            logger.error(f"유사 질문 검색 오류 (키워드): {e}", exc_info=True)
            return []

