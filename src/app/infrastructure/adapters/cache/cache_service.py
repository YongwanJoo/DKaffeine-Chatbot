import hashlib
import json
from typing import Optional, Dict, Any, List, Tuple
import os
import logging
import numpy as np

from app.domain.ports.cache_port import CachePort
from app.infrastructure.config.config_loader import get_config, get_config_int, get_config_float

logger = logging.getLogger(__name__)

class CacheService(CachePort):
    """캐시 서비스 (Redis 연결 가능 구조, Connection Pool 사용)
    
    챗봇 응답을 캐싱하여 동일한 질문에 대한 빠른 응답을 제공합니다.
    Redis를 사용하면 분산 환경에서도 캐시를 공유할 수 있습니다.
    
    특징:
    - Redis 또는 인메모리 캐시 지원
    - Connection Pool 사용 (성능 최적화)
    - 출처 정보 포함 캐싱
    - TTL 기반 자동 만료
    
    Example:
        ```python
        cache = CacheService(use_redis=True)
        
        # 저장 (출처 정보 포함)
        cache.set("company_001", "질문", "답변", sources=["FAQ"])
        
        # 조회
        answer = cache.get("company_001", "질문")
        
        # 조회 (출처 정보 포함)
        data = cache.get_with_sources("company_001", "질문")
        ```
    
    Args:
        use_redis: Redis 사용 여부 (기본값: False, 인메모리 사용)
    """
    
    def __init__(self, use_redis: bool = False, enable_semantic_cache: bool = True):
        self.use_redis = use_redis
        self.kb_version = get_config("KB_VERSION", "v1.0", section="cache")
        self.redis_pool = None
        self.redis_client = None
        self.enable_semantic_cache = enable_semantic_cache
        
        # 시맨틱 캐시를 위한 임베딩 모델 초기화
        self.embeddings = None
        self.semantic_similarity_threshold = get_config_float("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.95, section="cache")
        
        if enable_semantic_cache:
            try:
                from app.infrastructure.adapters.llm import LLMProvider
                self.embeddings = LLMProvider().get_embeddings()
                logger.info("시맨틱 캐시 활성화: 임베딩 모델 초기화 완료")
            except Exception as e:
                logger.warning(f"시맨틱 캐시 초기화 실패: {e}. Exact Match만 사용합니다.")
                self.enable_semantic_cache = False
        
        if use_redis:
            try:
                import redis
                from redis.connection import ConnectionPool
                
                # Connection Pool 생성
                # 배포 환경: 환경변수 REDIS_HOST, REDIS_PORT 등 또는 .secrets.toml 사용
                # 우선순위: 환경변수 > .secrets.toml > 기본값 (로컬 개발용)
                self.redis_pool = ConnectionPool(
                    host=get_config("host", "localhost", section="redis"),
                    port=get_config_int("port", 6379, section="redis"),
                    db=get_config_int("db", 0, section="redis"),
                    max_connections=get_config_int("max_connections", 50, section="redis"),
                    decode_responses=True,
                    retry_on_timeout=True,
                    health_check_interval=30  # 30초마다 연결 체크
                )
                self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                # 연결 테스트
                self.redis_client.ping()
                logger.info(
                    f"Redis connection pool initialized: "
                    f"max_connections={get_config_int('max_connections', 50, section='redis')}"
                )
            except ImportError:
                logger.warning("Redis not installed. Using in-memory cache.")
                self.use_redis = False
                self.cache = {}
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self.use_redis = False
                self.cache = {}
        else:
            # 인메모리 딕셔너리
            self.cache = {}
    
    def _compute_settings_hash(self, chatbot_settings: Optional[Dict[str, Any]]) -> str:
        """설정 해시 계산"""
        if not chatbot_settings:
            return "default"
        
        try:
            # 딕셔너리를 정렬된 JSON 문자열로 변환하여 해시 일관성 보장
            settings_str = json.dumps(chatbot_settings, sort_keys=True, default=str, ensure_ascii=False)
            hash_val = hashlib.sha256(settings_str.encode()).hexdigest()
            return hash_val
        except Exception as e:
            logger.warning(f"설정 해시 생성 실패 (기본값 사용): {e}")
            return "default"

    def _make_key(self, user_message: str, chatbot_settings: Optional[Dict[str, Any]] = None) -> str:
        """캐시 키 생성"""
        normalized = user_message.strip().lower()
        
        # 기본 해시 내용 (메시지)
        key_content = normalized
        
        # 설정이 있으면 해시에 포함 (설정 변경 시 캐시 무효화 효과 - Exact Match용)
        if chatbot_settings:
            settings_hash = self._compute_settings_hash(chatbot_settings)
            key_content += f":{settings_hash}"
        
        hash_val = hashlib.sha256(key_content.encode()).hexdigest()
        return f"cache:{self.kb_version}:{hash_val}"
    
    def get(self, user_message: str, chatbot_settings: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """캐시 조회 (하위 호환성을 위해 문자열만 반환)"""
        key = self._make_key(user_message, chatbot_settings)
        
        if self.use_redis:
            cached = self.redis_client.get(key)
        else:
            cached = self.cache.get(key)
        
        if cached:
            # JSON 형태인지 확인
            try:
                data = json.loads(cached)
                if isinstance(data, dict) and "answer" in data:
                    return data["answer"]
            except (json.JSONDecodeError, TypeError):
                pass
            # 문자열이면 그대로 반환 (하위 호환성)
            return cached
        return None
    
    def get_with_sources(self, user_message: str, chatbot_settings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """캐시 조회 (답변과 출처 정보 포함) - 시맨틱 캐시 지원"""
        # 1. Exact Match 먼저 시도
        key = self._make_key(user_message, chatbot_settings)
        
        if self.use_redis:
            cached = self.redis_client.get(key)
        else:
            cached = self.cache.get(key)
        
        if cached:
            try:
                data = json.loads(cached)
                if isinstance(data, dict):
                    logger.debug(f"Exact cache hit: {key[:50]}...")
                    return data
            except (json.JSONDecodeError, TypeError):
                # 문자열이면 하위 호환성을 위해 변환
                return {"answer": cached, "sources": None}
        
        # 2. 시맨틱 캐시 검색 (Exact Match 실패 시)
        if self.enable_semantic_cache and self.embeddings:
            semantic_result = self._get_semantic_cache(user_message, chatbot_settings)
            if semantic_result:
                logger.info(f"Semantic cache hit: similarity={semantic_result.get('similarity', 0):.3f}")
                return {
                    "answer": semantic_result.get("answer"),
                    "sources": semantic_result.get("sources")
                }
        
        return None
    
    def _get_semantic_cache(self, user_message: str, chatbot_settings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """시맨틱 캐시 검색: 유사한 질문의 캐시된 답변 반환
        
        성능 최적화:
        - 최대 검색 개수 제한 (기본 50개)
        - 임베딩이 있는 항목만 검색
        - 조기 종료: 높은 유사도 발견 시 즉시 반환
        """
        try:
            from app.infrastructure.utils.utils_math import cosine_similarity
            
            # 성능 최적화: 최대 검색 개수 제한
            max_search_count = get_config_int("SEMANTIC_CACHE_MAX_SEARCH", 50, section="cache")
            early_exit_threshold = get_config_float("SEMANTIC_CACHE_EARLY_EXIT", 0.98, section="cache")
            
            # 사용자 질문 임베딩 생성
            user_embedding = np.array(self.embeddings.embed_query(user_message))
            
            # Redis 또는 인메모리 캐시에서 모든 키 조회
            cache_keys = []
            if self.use_redis:
                # Redis에서 kb_version에 해당하는 모든 키 조회
                pattern = f"cache:{self.kb_version}:*"
                cache_keys = list(self.redis_client.scan_iter(match=pattern))
            else:
                # 인메모리 캐시에서 키 조회
                prefix = f"cache:{self.kb_version}:"
                cache_keys = [key for key in self.cache.keys() if key.startswith(prefix)]
            
            if not cache_keys:
                return None
            
            # 성능 최적화: 검색 개수 제한
            if len(cache_keys) > max_search_count:
                logger.debug(f"시맨틱 캐시: {len(cache_keys)}개 중 {max_search_count}개만 검색")
                cache_keys = cache_keys[:max_search_count]
            
            # 각 캐시 항목의 임베딩과 비교
            best_match = None
            best_similarity = 0.0
            checked_count = 0
            
            # 현재 요청의 설정 해시 계산
            current_settings_hash = self._compute_settings_hash(chatbot_settings)
            
            for cache_key in cache_keys:
                try:
                    # 캐시 데이터 조회
                    if self.use_redis:
                        cached_data = self.redis_client.get(cache_key)
                    else:
                        cached_data = self.cache.get(cache_key)
                    
                    if not cached_data:
                        continue
                    
                    # 캐시 데이터 파싱
                    try:
                        cache_data = json.loads(cached_data)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    
                    # 임베딩이 저장되어 있는지 확인
                    if "embedding" not in cache_data:
                        # 임베딩이 없으면 스킵 (기존 캐시는 임베딩이 없을 수 있음)
                        continue
                    
                    # 설정 해시 확인 (설정이 다르면 스킵)
                    cached_settings_hash = cache_data.get("settings_hash", "default")
                    
                    # 설정 해시가 다르면 스킵 (단, 둘 다 default인 경우는 허용)
                    if cached_settings_hash != current_settings_hash:
                        # logger.debug(f"시맨틱 캐시: 설정 불일치 (cached={cached_settings_hash[:8]}, current={current_settings_hash[:8]})")
                        continue
                    
                    checked_count += 1
                    cached_embedding = np.array(cache_data["embedding"])
                    
                    # 코사인 유사도 계산
                    similarity = cosine_similarity(user_embedding, cached_embedding)
                    
                    if similarity > best_similarity and similarity >= self.semantic_similarity_threshold:
                        best_similarity = similarity
                        best_match = {
                            "answer": cache_data.get("answer"),
                            "sources": cache_data.get("sources"),
                            "similarity": similarity
                        }
                        
                        # 성능 최적화: 매우 높은 유사도 발견 시 조기 종료
                        if similarity >= early_exit_threshold:
                            logger.debug(f"시맨틱 캐시: 조기 종료 (유사도={similarity:.3f} >= {early_exit_threshold})")
                            break
                
                except Exception as e:
                    logger.debug(f"시맨틱 캐시 검색 중 오류 (키: {cache_key[:50]}...): {e}")
                    continue
            
            if best_match:
                logger.info(f"시맨틱 캐시 히트: 유사도={best_similarity:.3f}, 검색한 항목={checked_count}개")
            
            return best_match
            
        except Exception as e:
            logger.warning(f"시맨틱 캐시 검색 실패: {e}")
            return None
    
    def set(self, user_message: str, answer: str, sources: Optional[list] = None, ttl: int = 3600, chatbot_settings: Optional[Dict[str, Any]] = None, rag_category: Optional[str] = None):
        """캐시 저장 (출처 정보, 카테고리 및 임베딩 포함)"""
        key = self._make_key(user_message, chatbot_settings)
        
        # 출처 정보와 카테고리 정보와 함께 저장
        cache_data = {
            "answer": answer,
            "sources": sources,
            "rag_category": rag_category,  # 카테고리 정보 추가
            "settings_hash": self._compute_settings_hash(chatbot_settings)
        }
        
        # 시맨틱 캐시를 위해 임베딩도 저장
        if self.enable_semantic_cache and self.embeddings:
            try:
                embedding = self.embeddings.embed_query(user_message)
                cache_data["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            except Exception as e:
                logger.warning(f"임베딩 생성 실패 (시맨틱 캐시 저장 스킵): {e}")
        
        cache_value = json.dumps(cache_data, ensure_ascii=False)
        
        if self.use_redis:
            self.redis_client.setex(key, ttl, cache_value)
        else:
            self.cache[key] = cache_value
    
    def clear(self):
        """캐시 초기화 (테스트용)"""
        if self.use_redis:
            self.redis_client.flushdb()
        else:
            self.cache.clear()

    def get_raw(self, key: str) -> Optional[str]:
        """Raw 키로 캐시 조회 (해싱 없음)"""
        if self.use_redis:
            return self.redis_client.get(key)
        else:
            return self.cache.get(key)
    
    def set_raw(self, key: str, value: str, ttl: int = 3600) -> None:
        """Raw 키로 캐시 저장 (해싱 없음)"""
        if self.use_redis:
            self.redis_client.setex(key, ttl, value)
        else:
            self.cache[key] = value





