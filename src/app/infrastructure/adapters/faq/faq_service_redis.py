"""FAQ 서비스 (Redis 기반) - Cosine Similarity 기반"""
from typing import Tuple, Optional
import json
import logging
import numpy as np
from pathlib import Path

from app.domain.ports.faq_port import FAQPort
from app.infrastructure.config.config_loader import get_config, get_config_int
from app.infrastructure.adapters.cache import HybridCache
from app.infrastructure.utils.utils_math import cosine_similarity

logger = logging.getLogger(__name__)


def normalize_question_for_search(question: str) -> str:
    """FAQ 검색을 위한 질문 정규화
    
    사용자 질문에서 요청 표현("알려줘", "알려주세요" 등)을 제거하여
    FAQ 질문과의 의미적 유사도를 더 정확하게 계산합니다.
    
    Args:
        question: 원본 질문
        
    Returns:
        정규화된 질문
    """
    import re
    
    request_patterns = [
        r'알려\s*줘\s*',
        r'알려\s*주세요\s*',
        r'알려\s*주시겠어요\s*',
        r'알려\s*주시겠습니까\s*',
        r'알려\s*주실\s*수\s*있나요\s*',
        r'알려\s*주실\s*수\s*있어요\s*',
        r'알려\s*주실\s*수\s*있습니까\s*',
        r'어떻게\s*해\s*',
        r'어떻게\s*해야\s*',
        r'어떻게\s*해야\s*하나요\s*',
        r'어떻게\s*하나요\s*',
        r'어떻게\s*하나\s*',
        r'어떻게\s*하는\s*',
        r'어떻게\s*하는지\s*',
        r'어떻게\s*하는지\s*알려\s*',
        r'\s*야\s*하나요\s*$',
        r'방법\s*알려\s*',
        r'방법\s*알려줘\s*',
        r'방법\s*알려주세요\s*',
        r'설명\s*해\s*줘\s*',
        r'설명\s*해\s*주세요\s*',
        r'설명\s*해\s*주시겠어요\s*',
        r'가르쳐\s*줘\s*',
        r'가르쳐\s*주세요\s*',
        r'말해\s*줘\s*',
        r'말해\s*주세요\s*',
        r'말씀\s*해\s*주세요\s*',
        r'\s*하나요\s*$',
        r'\s*인가요\s*$',
        r'\s*인지\s*$',
        r'\s*인지요\s*$',
        r'\s*야\s*하나요\s*$',
        # 추가: "뭐야", "뭐예요" 등 질문 표현 정규화
        r'\s*뭐야\s*$',
        r'\s*뭐예요\s*$',
        r'\s*뭐야요\s*$',
        r'\s*뭐니\s*$',
        r'\s*뭐냐\s*$',
        r'\s*뭔가\s*$',
        r'\s*뭔가요\s*$',
        r'\s*뭔지\s*$',
        r'\s*뭔지요\s*$',
        # 추가: "이란", "이란?" 등 질문 표현 정규화
        r'\s*이란\s*$',
        r'\s*이란\s*\?',
        r'\s*이란\s*요\s*$',
        r'\s*이란\s*가\s*$',
    ]
    
    normalized = question.strip()
    
    for pattern in request_patterns:
        normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
    
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


class FAQServiceRedis(FAQPort):
    """FAQ 서비스 (JSON 파일 또는 Redis 지원) - Cosine Similarity 기반
    
    FAQ 검색 서비스로, Cosine Similarity를 사용하여 의미 기반 검색을 수행합니다.
    임베딩 기반 검색이 실패할 경우 키워드 매칭으로 자동 폴백합니다.
    
    특징:
    - Cosine Similarity 기반 의미 검색
    - 배치 임베딩 생성 (성능 최적화)
    - LRU + TTL 하이브리드 캐시
    - Redis 또는 JSON 파일 지원
    
    Example:
        ```python
        service = FAQServiceRedis(use_redis=True)
        match, answer, score = service.search(
            company_id="company_001",
            user_message="휴가 신청 방법",
            threshold=0.7
        )
        ```
    
    Args:
        faq_path: FAQ JSON 파일 경로 (기본값: "./config/faq.json")
        use_redis: Redis 사용 여부 (기본값: False)
    """
    
    def __init__(self, faq_path: str = "./config/faq.json", use_redis: bool = False):
        self.faq_path = Path(faq_path)
        self.use_redis = use_redis
        self.faq_data = None
        self.redis_client = None
        
        try:
            from app.infrastructure.adapters.llm import LLMProvider
            self.embeddings = LLMProvider().get_embeddings()
            max_cache_size = get_config_int("FAQ_EMBEDDING_CACHE_SIZE", 1000, section="faq")
            cache_ttl = get_config_int("FAQ_EMBEDDING_CACHE_TTL", 3600, section="faq")
            self._faq_embeddings_cache = HybridCache(
                max_size=max_cache_size,
                ttl_seconds=cache_ttl
            )
            logger.info(f"FAQ 임베딩 초기화 완료: model=amazon.titan-embed-text-v2:0, cache_size={max_cache_size}, ttl={cache_ttl}s")
        except Exception as e:
            logger.warning(f"임베딩 모델 초기화 실패: {e}. 키워드 매칭으로 폴백합니다.")
            self.embeddings = None
            self._faq_embeddings_cache = None
        
        if use_redis:
            try:
                import redis
                from redis.connection import ConnectionPool
                
                self.redis_pool = ConnectionPool(
                    host=get_config("host", "localhost", section="redis"),
                    port=get_config_int("port", 6379, section="redis"),
                    db=get_config_int("db", 0, section="redis"),
                    max_connections=get_config_int("max_connections", 50, section="redis"),
                    decode_responses=True,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                self.redis_client.ping()
                logger.info(
                    f"✅ FAQServiceRedis: Redis connection pool initialized "
                    f"(max_connections={get_config_int('max_connections', 50, section='redis')})"
                )
            except ImportError:
                logger.warning("⚠️ FAQServiceRedis: Redis 패키지가 설치되지 않았습니다. JSON 파일을 사용합니다.")
                self.use_redis = False
            except Exception as e:
                logger.warning(f"⚠️ FAQServiceRedis: Redis 연결 실패 ({e}). JSON 파일을 사용합니다.")
                self.use_redis = False
        
        if not self.use_redis:
            self.faq_data = self._load_faq()
    
    def _load_faq(self) -> dict:
        """FAQ 데이터 로드 (JSON 파일)"""
        if not self.faq_path.exists():
            return {
                "default": [
                    {
                        "question": "휴가 신청 방법",
                        "answer": "휴가는 사내 포털 > 근태관리 > 휴가신청에서 신청할 수 있습니다.",
                        "keywords": ["휴가", "신청", "방법"]
                    },
                    {
                        "question": "출장 신청 절차",
                        "answer": "출장은 부서장 승인 후 사내 포털에서 신청하세요.",
                        "keywords": ["출장", "신청", "절차"]
                    }
                ]
            }
        
        with open(self.faq_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def search(
        self, 
        user_message: str, 
        threshold: float = 0.7,
        user_embedding: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """FAQ 검색 (DB 실시간 조회)
        
        사용자 메시지와 DB의 활성 FAQ를 Cosine Similarity로 비교하여
        가장 유사한 FAQ를 찾습니다.
        
        Args:
            user_message: 사용자 질문
            threshold: 유사도 임계값 (0.0 ~ 1.0, 기본값: 0.7)
            user_embedding: 사용자 질문 임베딩 (제공되면 재사용, None이면 생성)
        
        Returns:
            (faq_match, faq_answer, faq_confidence) 튜플
        """
        # Redis 사용 여부와 관계없이 DB 실시간 조회를 우선 사용 (사용자 요구사항)
        return self._search_db(user_message, threshold, user_embedding)

    def _search_db(self, user_message: str, threshold: float, user_embedding: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str], Optional[float]]:
        """DB에서 FAQ 검색 (Cosine Similarity 기반)"""
        logger.info(f"[FAQ DB 검색] query='{user_message}'")
        
        try:
            from app.infrastructure.persistence.session import db_session
            from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
            
            # 임베딩 모델이 없으면 키워드 검색으로 폴백 (DB 기반)
            if not self.embeddings or self._faq_embeddings_cache is None:
                return self._search_db_keyword_fallback(user_message, threshold)

            normalized_user_message = normalize_question_for_search(user_message)
            
            if user_embedding is None:
                user_embedding = np.array(self.embeddings.embed_query(normalized_user_message))
            else:
                user_embedding = np.array(user_embedding)

            with db_session() as db:
                repo = FaqRepository(db)
                # 활성 FAQ 모두 조회
                active_faqs = repo.find_all_active(limit=1000)
                
                if not active_faqs:
                    logger.info("[FAQ DB 검색] 활성 FAQ가 없습니다.")
                    return False, None, None
                
                best_match = None
                best_score = 0.0
                
                for faq in active_faqs:
                    # 캐시 키 생성 (ID와 질문 기준)
                    cache_key = f"faq_embedding:db:{faq.id}:{faq.question}"
                    
                    cached_embedding = self._faq_embeddings_cache.get(cache_key)
                    if cached_embedding is None:
                        try:
                            # 임베딩 생성 및 캐시
                            faq_embedding = np.array(self.embeddings.embed_query(faq.question))
                            self._faq_embeddings_cache.set(cache_key, faq_embedding)
                        except Exception as e:
                            logger.warning(f"FAQ 임베딩 생성 실패 (ID={faq.id}): {e}")
                            continue
                    else:
                        faq_embedding = cached_embedding
                    
                    # 유사도 계산
                    score = cosine_similarity(user_embedding, faq_embedding)
                    
                    if score > best_score:
                        best_score = score
                        best_match = faq
                
                if best_score >= threshold and best_match:
                    logger.info(f"[FAQ DB 매칭 성공] 질문='{best_match.question}', 점수={best_score:.3f}")
                    return True, best_match.answer, best_score
                
                logger.info(f"[FAQ DB 매칭 실패] 최고점수={best_score:.3f}, 임계값={threshold:.2f}. 키워드 검색으로 폴백합니다.")
                return self._search_db_keyword_fallback(user_message, threshold)

        except Exception as e:
            logger.error(f"FAQ DB 검색 오류: {e}", exc_info=True)
            return False, None, None

    def _search_db_keyword_fallback(self, user_message: str, threshold: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """DB에서 FAQ 검색 (키워드 매칭 폴백)"""
        try:
            from app.infrastructure.persistence.session import db_session
            from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
            
            normalized_user_message = normalize_question_for_search(user_message)
            keywords = [w for w in normalized_user_message.split() if len(w) > 1]
            
            if not keywords:
                return False, None, None
                
            with db_session() as db:
                repo = FaqRepository(db)
                # 키워드로 검색
                candidates = repo.find_by_keywords(keywords, limit=10)
                
                if not candidates:
                    return False, None, None
                
                # 개선된 점수 계산 (더 관대하게)
                best_match = None
                best_score = 0.0
                
                for faq in candidates:
                    faq_question_lower = faq.question.lower()
                    faq_answer_lower = faq.answer.lower()
                    target_text = f"{faq_question_lower} {faq_answer_lower}"
                    
                    # 1. 질문에 매칭된 키워드 수
                    question_matches = sum(1 for kw in keywords if kw.lower() in faq_question_lower)
                    # 2. 답변에 매칭된 키워드 수
                    answer_matches = sum(1 for kw in keywords if kw.lower() in faq_answer_lower)
                    # 3. 전체 텍스트에 매칭된 키워드 수
                    total_matches = sum(1 for kw in keywords if kw.lower() in target_text)
                    
                    # 점수 계산: 질문 매칭에 더 높은 가중치
                    question_score = question_matches / len(keywords) if keywords else 0.0
                    answer_score = answer_matches / len(keywords) if keywords else 0.0
                    total_score = total_matches / len(keywords) if keywords else 0.0
                    
                    # 최종 점수: 질문 매칭에 더 높은 가중치 부여
                    score = max(
                        question_score * 1.2,  # 질문 매칭에 더 높은 가중치
                        answer_score * 0.8,
                        total_score
                    )
                    
                    # 핵심 키워드가 많이 매칭되면 추가 보너스
                    # 질문에 모든 키워드가 매칭되면 높은 점수 부여
                    if question_matches == len(keywords) and len(keywords) >= 2:
                        score = max(score, 0.7)  # 모든 키워드 매칭 시 높은 점수
                    elif question_matches >= 2 and len(keywords) >= 2:
                        score = max(score, 0.5)  # 핵심 키워드 매칭 시 최소 0.5 보장
                    
                    if score > best_score:
                        best_score = score
                        best_match = faq
                
                # 키워드 매칭 임계값 고정 (0.6)
                keyword_threshold = 0.6
                
                if best_score >= keyword_threshold and best_match:
                    logger.info(f"[FAQ DB 키워드 매칭 성공] 질문='{best_match.question}', 점수={best_score:.2f}, 임계값={keyword_threshold:.2f}")
                    return True, best_match.answer, best_score
                
                logger.info(f"[FAQ DB 키워드 매칭 실패] 최고점수={best_score:.2f}, 임계값={keyword_threshold:.2f}")
                return False, None, None
        except Exception as e:
            logger.error(f"FAQ DB 키워드 검색 오류: {e}", exc_info=True)
            return False, None, None
    
    def _search_redis(self, user_message: str, threshold: float, user_embedding: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str], Optional[float]]:
        """Redis에서 FAQ 검색 (Cosine Similarity 기반)
        
        Args:
            user_message: 사용자 질문
            threshold: 유사도 임계값
            user_embedding: 사용자 질문 임베딩 (제공되면 재사용, None이면 생성)
        """
        company_id = "default"  # company_id 제거: 기본값 사용
        logger.info(f"[FAQ Redis 검색] query='{user_message}'")
        
        if not self.embeddings or self._faq_embeddings_cache is None:
            return self._search_redis_keyword_fallback(user_message, threshold)
        
        try:
            normalized_user_message = normalize_question_for_search(user_message)
            if normalized_user_message != user_message:
                logger.debug(f"[FAQ 질문 정규화] 원본='{user_message}' -> 정규화='{normalized_user_message}'")
            
            if user_embedding is None:
                user_embedding = np.array(self.embeddings.embed_query(normalized_user_message))
            else:
                user_embedding = np.array(user_embedding)
            import re
            normalized_message = normalized_user_message.lower()
            particles = ['에', '에서', '에게', '을', '를', '이', '가', '은', '는', '와', '과', '도', '로', '으로', '의', '에 대해', '에 관한']
            cleaned_message = normalized_message
            for particle in particles:
                cleaned_message = cleaned_message.replace(particle, ' ')
            cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()
            
            candidate_keys = set()
            search_texts = [normalized_message, cleaned_message]
            
            company_ids_to_search = [company_id]
            if company_id != "default":
                company_ids_to_search.append("default")
            
            for search_company_id in company_ids_to_search:
                for search_text in search_texts:
                    for word in search_text.split():
                        if len(word) > 1:
                            keyword_index_key = f"faq_index:{search_company_id}:{word}"
                            keys = self.redis_client.smembers(keyword_index_key)
                            candidate_keys.update(keys)
            
            if not candidate_keys:
                for search_company_id in company_ids_to_search:
                    company_faq_list_key = f"faq_list:{search_company_id}"
                    keys = self.redis_client.smembers(company_faq_list_key)
                    candidate_keys.update(keys)
                    if candidate_keys:
                        break
            
            if not candidate_keys:
                logger.info(f"[FAQ Redis 검색] 후보 없음 (company_id={company_id}, 검색한 company_ids={company_ids_to_search})")
                return False, None, None
            
            faq_data_list = []
            faq_metadata = []
            
            for faq_key in candidate_keys:
                faq_json = self.redis_client.get(faq_key)
                if not faq_json:
                    continue
                
                try:
                    faq = json.loads(faq_json)
                    question = faq.get("question", "")
                    answer = faq.get("answer", "")
                    cache_key = f"{faq_key}_{question}"
                    
                    embedding_key = None
                    redis_embedding_json = None
                    for search_company_id in company_ids_to_search:
                        embedding_key = f"faq_embedding:{search_company_id}:{question}"
                        redis_embedding_json = self.redis_client.get(embedding_key)
                        if redis_embedding_json:
                            break
                    
                    if redis_embedding_json:
                        try:
                            redis_embedding = json.loads(redis_embedding_json)
                            faq_embedding = np.array(redis_embedding)
                            self._faq_embeddings_cache.set(cache_key, faq_embedding)
                            faq_metadata.append((faq_key, cache_key, faq, faq_embedding))
                            continue
                        except Exception as e:
                            logger.warning(f"Redis 임베딩 파싱 실패: {e}, 인메모리 캐시 확인")
                    
                    cached_embedding = self._faq_embeddings_cache.get(cache_key)
                    if cached_embedding is not None:
                        faq_metadata.append((faq_key, cache_key, faq, cached_embedding))
                    else:
                        try:
                            logger.info(f"FAQ 임베딩 온더플라이 생성: question='{question[:50]}...'")
                            faq_embedding = np.array(self.embeddings.embed_query(question))
                            
                            try:
                                embedding_json = json.dumps(faq_embedding.tolist(), ensure_ascii=False)
                                primary_embedding_key = f"faq_embedding:{company_id}:{question}"
                                self.redis_client.set(primary_embedding_key, embedding_json)
                                logger.debug(f"FAQ 임베딩 Redis 저장 완료: {primary_embedding_key}")
                            except Exception as e:
                                logger.warning(f"FAQ 임베딩 Redis 저장 실패 (계속 진행): {e}")
                            
                            self._faq_embeddings_cache.set(cache_key, faq_embedding)
                            faq_metadata.append((faq_key, cache_key, faq, faq_embedding))
                            logger.debug(f"FAQ 임베딩 온더플라이 생성 완료: question='{question[:50]}...'")
                        except Exception as e:
                            logger.warning(
                                f"FAQ 임베딩 온더플라이 생성 실패 (스킵): "
                                f"question='{question[:50]}...', error={e}"
                            )
                            continue
                        
                except json.JSONDecodeError:
                    continue
            
            if not faq_metadata:
                logger.warning(
                    f"[FAQ Redis 검색] 모든 FAQ의 임베딩 생성/로드 실패, "
                    f"키워드 폴백으로 전환: query='{user_message[:50]}...'"
                )
                return self._search_redis_keyword_fallback(user_message, threshold)
            
            best_match = None
            best_score = 0.0
            all_scores = []
            
            for faq_key, cache_key, faq, faq_embedding in faq_metadata:
                if faq_embedding is None:
                    continue
                
                try:
                    score = cosine_similarity(user_embedding, faq_embedding)
                    all_scores.append((faq.get("question", "unknown"), score))
                    
                    if score > best_score:
                        best_score = score
                        best_match = faq
                except Exception as e:
                    logger.warning(f"Cosine similarity 계산 실패: {e}")
                    continue
            
            if all_scores:
                all_scores.sort(key=lambda x: x[1], reverse=True)
                top_scores = all_scores[:3]
                logger.debug(f"[FAQ 유사도 점수] 질문='{user_message[:30]}...', 상위 3개: {[(q[:30], f'{s:.3f}') for q, s in top_scores]}")
            
            if best_score >= threshold and best_match:
                logger.info(f"[FAQ Redis 매칭 성공] 질문='{best_match.get('question', 'unknown')}', 점수={best_score:.3f}, 임계값={threshold:.2f}")
                return True, best_match["answer"], best_score
            
            logger.info(f"[FAQ Redis 매칭 실패] 최고점수={best_score:.3f}, 임계값={threshold:.2f}, 후보={len(candidate_keys)}개. 키워드 검색으로 폴백합니다.")
            return self._search_redis_keyword_fallback(user_message, threshold)
            
        except Exception as e:
            logger.error(f"FAQ Redis 검색 오류 (Cosine): {e}", exc_info=True)
            return self._search_redis_keyword_fallback(user_message, threshold)
    
    def _search_redis_keyword_fallback(self, user_message: str, threshold: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """Redis에서 FAQ 검색 (키워드 매칭 폴백)"""
        company_id = "default"  # company_id 제거: 기본값 사용
        logger.info(f"[FAQ Redis 검색] 키워드 매칭 폴백 사용")
        normalized_user_message = normalize_question_for_search(user_message)
        normalized_message = normalized_user_message.lower()
        
        import re
        particles = ['에', '에서', '에게', '을', '를', '이', '가', '은', '는', '와', '과', '도', '로', '으로', '의', '에 대해', '에 관한']
        cleaned_message = normalized_message
        for particle in particles:
            cleaned_message = cleaned_message.replace(particle, ' ')
        cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()
        
        candidate_keys = set()
        search_texts = [normalized_message, cleaned_message]
        
        company_ids_to_search = [company_id]
        
        for search_company_id in company_ids_to_search:
            for search_text in search_texts:
                for word in search_text.split():
                    if len(word) > 1:
                        keyword_index_key = f"faq_index:{search_company_id}:{word}"
                        keys = self.redis_client.smembers(keyword_index_key)
                        candidate_keys.update(keys)
        
        if not candidate_keys:
            for search_company_id in company_ids_to_search:
                company_faq_list_key = f"faq_list:{search_company_id}"
                keys = self.redis_client.smembers(company_faq_list_key)
                candidate_keys.update(keys)
                if candidate_keys:
                    break
        
        best_match = None
        best_score = 0.0
        
        for faq_key in candidate_keys:
            faq_json = self.redis_client.get(faq_key)
            if not faq_json:
                continue
            
            try:
                faq = json.loads(faq_json)
                faq_question = faq.get("question", "").lower()
                faq_answer = faq.get("answer", "").lower()
                keywords = faq.get("keywords", [])
                
                # 1. 키워드 매칭 점수
                keyword_matches = sum(1 for kw in keywords if kw in normalized_message or kw in cleaned_message)
                keyword_score = 0.0
                if keywords:
                    keyword_score = keyword_matches / len(keywords)
                    # 키워드가 하나라도 매칭되면 최소 점수 보장
                    if keyword_matches > 0 and keyword_score < 0.3:
                        keyword_score = 0.3 + (keyword_matches - 1) * 0.15
                
                # 2. FAQ 질문/답변에 사용자 질문의 핵심 단어가 포함되는지 확인
                user_words = set([w for w in cleaned_message.split() if len(w) > 1])
                faq_text = f"{faq_question} {faq_answer}"
                text_matches = sum(1 for word in user_words if word in faq_text)
                text_score = text_matches / len(user_words) if user_words else 0.0
                
                # 3. FAQ 질문에 사용자 질문의 핵심 단어가 많이 포함되면 추가 점수
                question_matches = sum(1 for word in user_words if word in faq_question)
                question_score = question_matches / len(user_words) if user_words else 0.0
                
                # 4. 최종 점수: 키워드 점수와 텍스트 매칭 점수를 결합 (더 관대하게)
                score = max(keyword_score, text_score * 0.8, question_score * 0.9)
                
                # 5. 핵심 단어가 많이 매칭되면 추가 보너스
                if text_matches >= 2 and len(user_words) >= 2:
                    score = max(score, 0.5)  # 최소 0.5 보장
                
                if score > best_score:
                    best_score = score
                    best_match = faq
            except json.JSONDecodeError:
                continue
        
        if best_score >= threshold and best_match:
            logger.info(f"[FAQ Redis 매칭 성공] 질문='{best_match.get('question', 'unknown')}', 점수={best_score:.2f}, 임계값={threshold:.2f}")
            return True, best_match["answer"], best_score
        
        logger.info(f"[FAQ Redis 매칭 실패] 최고점수={best_score:.2f}, 임계값={threshold:.2f}, 후보={len(candidate_keys)}개")
        return False, None, None
    
    def _search_json(self, user_message: str, threshold: float, user_embedding: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str], Optional[float]]:
        """JSON 파일에서 FAQ 검색 (키워드 매칭만 사용, user_embedding은 사용하지 않음)"""
        company_id = "default"  # company_id 제거: 기본값 사용
        faqs = self.faq_data.get(company_id, self.faq_data.get("default", []))
        
        normalized_message = user_message.lower()
        
        best_match = None
        best_score = 0.0
        
        for faq in faqs:
            keywords = faq.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in normalized_message)
            score = matches / len(keywords) if keywords else 0.0
            
            if score > best_score:
                best_score = score
                best_match = faq
        
        if best_score >= threshold:
            return True, best_match["answer"], best_score
        
        return False, None, None
    
    def warmup_faq_cache(self, company_id: str = "default", limit: int = 1000):
        """FAQ 캐시 워밍업: PostgreSQL의 활성 FAQ를 Redis에 미리 로드
        
        서버 시작 시 호출하여 자주 사용되는 FAQ를 Redis에 미리 로드합니다.
        이를 통해 Redis miss를 줄이고 응답 속도를 향상시킵니다.
        
        Args:
            company_id: 회사 ID (기본값: "default")
            limit: 최대 로드할 FAQ 개수 (기본값: 1000)
        
        Returns:
            (성공 개수, 실패 개수) 튜플
        """
        if not self.use_redis or not self.redis_client:
            logger.info("FAQ 워밍업 스킵: Redis가 비활성화되어 있습니다.")
            return 0, 0
        
        try:
            from app.infrastructure.persistence.session import db_session
            from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
            from app.infrastructure.persistence.models.faq_model import FaqStatus
            
            logger.info(f"FAQ 캐시 워밍업 시작: company_id={company_id}, limit={limit}")
            
            success_count = 0
            error_count = 0
            
            with db_session() as db:
                repo = FaqRepository(db)
                active_faqs = repo.find_all_active(limit=limit)  # Java 백엔드와 일치: company_id 파라미터 없음
                
                logger.info(f"FAQ 워밍업: {len(active_faqs)}개의 활성 FAQ 발견")
                
                for faq in active_faqs:
                    try:
                        self._replicate_single_faq_to_redis(faq, company_id)
                        success_count += 1
                        logger.debug(f"FAQ 워밍업 성공 (ID={faq.id}, question='{faq.question[:50]}...')")
                    except Exception as e:
                        error_count += 1
                        logger.warning(f"FAQ 워밍업 실패 (ID={faq.id}, question='{faq.question[:50]}...'): {e}", exc_info=True)
                        continue
            
            logger.info(
                f"FAQ 캐시 워밍업 완료: 성공={success_count}, 실패={error_count}, "
                f"총={len(active_faqs) if 'active_faqs' in locals() else 0}"
            )
            
            return success_count, error_count
            
        except Exception as e:
            logger.error(f"FAQ 캐시 워밍업 중 오류: {e}", exc_info=True)
            return 0, 0
    
    def _replicate_single_faq_to_redis(self, faq, company_id: str):
        """단일 FAQ 객체를 Redis에 복제 (워밍업용)
        
        Args:
            faq: Faq 모델 객체
            company_id: 회사 ID
        """
        question = str(faq.question) if faq.question else ""
        answer_text = str(faq.answer) if faq.answer else ""
        faq_id = faq.id  # Java 백엔드와 일치: id 사용
        
        if not question or not answer_text:
            raise ValueError(f"FAQ 데이터가 유효하지 않습니다 (ID={faq_id}, question={bool(question)}, answer={bool(answer_text)})")
        
        particles = ['에', '에서', '에게', '을', '를', '이', '가', '은', '는', '와', '과', '도', '로', '으로', '의', 
                    '에 대해', '에 관한', '시', '어떻게', '무엇인가요', '하나요', '인가요', '는가요', '인지', '인지요']
        
        combined_text = f"{question} {answer_text}".lower()
        for particle in particles:
            combined_text = combined_text.replace(particle, ' ')
        keywords = [word.strip() for word in combined_text.split() if len(word.strip()) > 1]
        keywords = sorted(list(set(keywords)))[:10]
        
        faq_data = {
            "question": question,
            "answer": answer_text,
            "keywords": keywords
        }
        
        faq_key = f"faq:{company_id}:{question}"
        faq_json = json.dumps(faq_data, ensure_ascii=False)
        self.redis_client.set(faq_key, faq_json)
        
        for keyword in keywords:
            keyword_index_key = f"faq_index:{company_id}:{keyword}"
            self.redis_client.sadd(keyword_index_key, faq_key)
        
        company_faq_list_key = f"faq_list:{company_id}"
        self.redis_client.sadd(company_faq_list_key, faq_key)
        
        if self.embeddings:
            try:
                embedding = self.embeddings.embed_query(question)
                embedding_key = f"faq_embedding:{company_id}:{question}"
                embedding_json = json.dumps(embedding, ensure_ascii=False)
                self.redis_client.set(embedding_key, embedding_json)
                logger.debug(f"FAQ 임베딩 사전 생성 완료: {embedding_key} (질문만 사용)")
            except Exception as e:
                logger.warning(f"FAQ 임베딩 사전 생성 실패 (ID={faq_id}, question='{question[:50]}...'): {e}")

