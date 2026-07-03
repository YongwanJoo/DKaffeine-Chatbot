"""질문 로그 서비스 (Redis + TTL 기반)

FAQ 빈도 분석을 위한 질문 로깅 서비스입니다.
Redis를 사용하여 질문 로그를 저장하고, TTL로 자동 만료됩니다.
"""
import hashlib
import json
import time
from typing import Optional, Dict, List, Any
import logging
from app.infrastructure.config.config_loader import get_config_int

logger = logging.getLogger(__name__)


class QuestionLogService:
    """질문 로그 서비스 (Redis 기반)
    
    FAQ 검색 실패 질문을 Redis에 로깅하여 빈도 기반 FAQ 자동 생성을 지원합니다.
    TTL로 자동 만료되어 오래된 데이터는 자동으로 정리됩니다.
    
    Redis 키 구조:
    - question_log:{company_id}:{question_hash} - 질문 로그 (Hash)
      - question: 질문 텍스트
      - frequency: 빈도 (정수)
      - embedding: 임베딩 벡터 (JSON, 선택적)
      - cluster_id: 클러스터 ID (배치 작업에서 할당)
      - created_at: 생성 시간 (Unix timestamp)
      - updated_at: 최종 업데이트 시간 (Unix timestamp)
    
    TTL: 기본 7일 (604800초) - 환경변수 QUESTION_LOG_TTL로 설정 가능
    """
    
    def __init__(self, redis_client=None, ttl_seconds: Optional[int] = None):
        """질문 로그 서비스 초기화
        
        Args:
            redis_client: Redis 클라이언트 (None이면 CacheService에서 가져옴)
            ttl_seconds: TTL 초 (None이면 환경변수 또는 기본값 7일 사용)
        """
        default_ttl = get_config_int("QUESTION_LOG_TTL", 604800, section="question_log")  # 기본 7일 (초)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else default_ttl
        self.redis_client = redis_client
        self._use_redis = redis_client is not None
    
    def _get_redis_client(self):
        """Redis 클라이언트 가져오기"""
        if self.redis_client:
            return self.redis_client
        
        # CacheService에서 Redis 클라이언트 가져오기
        try:
            from app.infrastructure.adapters.cache import CacheService
            cache_service = CacheService(use_redis=True)
            if cache_service.use_redis and cache_service.redis_client:
                return cache_service.redis_client
        except Exception as e:
            logger.warning(f"Redis 클라이언트 가져오기 실패: {e}")
        
        return None
    
    def _make_key(self, company_id: str, question: str) -> str:
        """Redis 키 생성"""
        normalized = question.strip().lower()
        question_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"question_log:{company_id}:{question_hash}"
    
    def log_question(
        self,
        question: str,
        company_id: str = "default",
        user_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        auto_generate_embedding: bool = True
    ) -> bool:
        """질문 로그 기록 (빈도 증가 또는 새 로그 생성)
        
        Args:
            question: 질문 텍스트
            company_id: 회사 ID
            user_id: 사용자 ID (선택적)
            embedding: 임베딩 벡터 (선택적)
            auto_generate_embedding: 임베딩이 없을 때 자동 생성 여부 (기본: True)
            
        Returns:
            성공 여부
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            logger.warning("Redis가 사용 불가능하여 질문 로그를 기록할 수 없습니다.")
            return False
        
        try:
            key = self._make_key(company_id, question)
            current_time = int(time.time())
            
            # 기존 로그 확인
            existing = redis_client.hgetall(key)
            
            # 임베딩이 없고 자동 생성이 활성화되어 있으면 생성
            if not embedding and auto_generate_embedding:
                try:
                    from app.infrastructure.adapters.llm import LLMProvider
                    llm_provider = LLMProvider()
                    embeddings_model = llm_provider.get_embeddings()
                    embedding = embeddings_model.embed_query(question)
                except Exception as e:
                    logger.debug(f"임베딩 자동 생성 실패 (무시하고 계속 진행): {e}")
                    embedding = None
            
            if existing:
                # 기존 로그의 빈도 증가
                frequency = int(existing.get("frequency", "1")) + 1
                update_data = {
                    "frequency": frequency,
                    "updated_at": current_time
                }
                
                # 임베딩이 있고 기존에 없으면 추가
                if embedding and not existing.get("embedding"):
                    update_data["embedding"] = json.dumps(embedding)
                
                redis_client.hset(key, mapping=update_data)
                redis_client.expire(key, self.ttl_seconds)  # TTL 갱신
                logger.debug(f"질문 로그 빈도 증가: question='{question[:50]}...', frequency={frequency}")
            else:
                # 새 로그 생성
                log_data = {
                    "question": question,
                    "frequency": "1",
                    "created_at": str(current_time),
                    "updated_at": str(current_time)
                }
                
                if user_id:
                    log_data["user_id"] = user_id
                
                if embedding:
                    log_data["embedding"] = json.dumps(embedding)
                
                redis_client.hset(key, mapping=log_data)
                redis_client.expire(key, self.ttl_seconds)  # TTL 설정
                logger.debug(f"질문 로그 생성: question='{question[:50]}...'")
            
            return True
            
        except Exception as e:
            logger.error(f"질문 로그 기록 실패: {e}", exc_info=True)
            return False
    
    def get_all_logs(
        self, 
        company_id: Optional[str] = None, 
        limit: int = 10000,
        min_frequency: int = 1,
        days_back: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """모든 질문 로그 조회 (배치 작업용)
        
        Args:
            company_id: 회사 ID (None이면 모든 회사)
            limit: 최대 조회 수
            min_frequency: 최소 빈도 (기본: 1)
            days_back: 며칠 전까지의 데이터만 조회 (None이면 전체)
            
        Returns:
            질문 로그 리스트
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return []
        
        try:
            pattern = f"question_log:{company_id}:*" if company_id else "question_log:*"
            keys = []
            
            # Redis SCAN으로 키 조회 (메모리 효율적)
            cursor = 0
            while True:
                cursor, batch_keys = redis_client.scan(cursor, match=pattern, count=1000)
                keys.extend(batch_keys)
                if cursor == 0 or len(keys) >= limit * 2:  # 필터링을 위해 더 많이 조회
                    break
            
            current_time = int(time.time())
            cutoff_time = current_time - (days_back * 24 * 60 * 60) if days_back else 0
            
            logs = []
            for key in keys:
                log_data = redis_client.hgetall(key)
                if not log_data:
                    continue
                
                frequency = int(log_data.get("frequency", "1"))
                if frequency < min_frequency:
                    continue
                
                created_at = int(log_data.get("created_at", "0"))
                if days_back and created_at < cutoff_time:
                    continue
                
                log_dict = {
                    "key": key,
                    "question": log_data.get("question", ""),
                    "frequency": frequency,
                    "created_at": created_at,
                    "updated_at": int(log_data.get("updated_at", "0")),
                }
                
                if "user_id" in log_data:
                    log_dict["user_id"] = log_data["user_id"]
                
                if "embedding" in log_data:
                    try:
                        log_dict["embedding"] = json.loads(log_data["embedding"])
                    except:
                        log_dict["embedding"] = None
                else:
                    log_dict["embedding"] = None
                
                if "cluster_id" in log_data:
                    log_dict["cluster_id"] = int(log_data["cluster_id"])
                else:
                    log_dict["cluster_id"] = None
                
                logs.append(log_dict)
            
            # 빈도 내림차순 정렬
            logs.sort(key=lambda x: x["frequency"], reverse=True)
            return logs[:limit]
            
        except Exception as e:
            logger.error(f"질문 로그 조회 실패: {e}", exc_info=True)
            return []
    
    def get_cluster_frequency(
        self,
        question: str,
        question_embedding: Optional[List[float]],
        company_id: str = "default",
        similarity_threshold: float = 0.8,
        days_back: int = 30
    ) -> int:
        """의미 기반 클러스터의 총 빈도수 계산
        
        현재 질문과 유사한 의미를 가진 질문들을 클러스터로 묶어서
        클러스터의 총 빈도수를 반환합니다.
        
        예시:
        - "연차 사용 어캐해?" (빈도: 5)
        - "연차 사용 방법에 대해서 알려줘" (빈도: 3)
        - "연차 사용 방법 내규에 대해 알려줘" (빈도: 2)
        → 총 빈도수: 10
        
        Args:
            question: 현재 질문
            question_embedding: 질문 임베딩 벡터 (None이면 자동 생성)
            company_id: 회사 ID
            similarity_threshold: 유사도 임계값 (기본: 0.8)
            days_back: 며칠 전까지의 데이터만 조회 (기본: 30일)
            
        Returns:
            클러스터의 총 빈도수 (유사한 질문들의 빈도수 합산)
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return 0
        
        try:
            # 모든 질문 로그 조회
            all_logs = self.get_all_logs(
                company_id=company_id,
                limit=10000,
                min_frequency=1,
                days_back=days_back
            )
            
            if not all_logs:
                logger.debug(f"질문 로그가 없음: question='{question[:50]}...'")
                return 0
            
            # 임베딩이 없으면 생성 시도
            question_embedding = None
            if not question_embedding:
                try:
                    from app.infrastructure.adapters.llm import LLMProvider
                    llm_provider = LLMProvider()
                    embeddings_model = llm_provider.get_embeddings()
                    question_embedding = embeddings_model.embed_query(question)
                    logger.debug(f"임베딩 생성 성공: question='{question[:50]}...'")
                except Exception as e:
                    logger.warning(f"⚠️ [FAQ] 임베딩 생성 실패 (텍스트 기반 매칭으로 폴백): {e}")
                    question_embedding = None  # 임베딩 없이도 텍스트 기반으로 진행
            
            import numpy as np
            from app.infrastructure.utils.utils_math import cosine_similarity
            
            # 유사한 질문들의 빈도수 합산
            total_frequency = 0
            similar_count = 0
            exact_match_count = 0
            
            for log in all_logs:
                log_question = log.get("question", "")
                log_embedding = log.get("embedding")
                frequency = log.get("frequency", 1)
                
                # 1. 정확한 질문 매칭 (임베딩 실패 시에도 작동)
                if log_question and log_question.strip() == question.strip():
                    total_frequency += frequency
                    exact_match_count += 1
                    logger.debug(f"정확한 질문 매칭: question='{log_question[:50]}...', frequency={frequency}")
                    continue
                
                # 2. 임베딩 기반 유사도 매칭 (임베딩이 있을 때만)
                if question_embedding and log_embedding:
                    try:
                        # 임베딩이 문자열이면 파싱
                        if isinstance(log_embedding, str):
                            import json
                            log_embedding = json.loads(log_embedding)
                        
                        log_embedding = np.array(log_embedding)
                        question_embedding_array = np.array(question_embedding)
                        
                        # 코사인 유사도 계산
                        similarity = cosine_similarity(question_embedding_array, log_embedding)
                        
                        if similarity >= similarity_threshold:
                            total_frequency += frequency
                            similar_count += 1
                            logger.debug(
                                f"유사 질문 발견: question='{log_question[:50]}...', "
                                f"frequency={frequency}, similarity={similarity:.3f}"
                            )
                    except Exception as e:
                        logger.debug(f"임베딩 비교 실패 (무시): {e}")
                        continue
                # 임베딩이 없는 로그는 스킵 (이미 정확한 매칭에서 처리됨)
                
                try:
                    # 임베딩이 문자열이면 파싱
                    if isinstance(log_embedding, str):
                        import json
                        log_embedding = json.loads(log_embedding)
                    
                    log_embedding = np.array(log_embedding)
                    
                    # 코사인 유사도 계산
                    similarity = cosine_similarity(question_embedding, log_embedding)
                    
                    if similarity >= similarity_threshold:
                        frequency = log.get("frequency", 1)
                        total_frequency += frequency
                        similar_count += 1
                        logger.debug(
                            f"유사 질문 발견: question='{log.get('question', '')[:50]}...', "
                            f"frequency={frequency}, similarity={similarity:.3f}"
                        )
                except Exception as e:
                    logger.debug(f"임베딩 비교 실패 (무시): {e}")
                    continue
            
            logger.info(
                f"🔍 [FAQ] 클러스터 빈도수 계산: question='{question[:50]}...', "
                f"정확한 매칭={exact_match_count}, 유사 질문 수={similar_count}, "
                f"총 빈도수={total_frequency}, 임계값={similarity_threshold}"
            )
            
            return total_frequency
            
        except Exception as e:
            logger.error(f"클러스터 빈도수 계산 실패: {e}", exc_info=True)
            return 0
    
    def update_cluster(self, keys: List[str], cluster_id: int):
        """로그에 클러스터 ID 할당
        
        Args:
            keys: Redis 키 리스트
            cluster_id: 클러스터 ID
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        
        try:
            for key in keys:
                redis_client.hset(key, "cluster_id", cluster_id)
                redis_client.expire(key, self.ttl_seconds)  # TTL 갱신
        except Exception as e:
            logger.error(f"클러스터 ID 업데이트 실패: {e}", exc_info=True)
    
    def mark_processed(self, keys: List[str]):
        """로그를 처리 완료로 표시 (선택적, 필요시 사용)
        
        Args:
            keys: Redis 키 리스트
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        
        try:
            for key in keys:
                redis_client.hset(key, "processed", "1")
                redis_client.expire(key, self.ttl_seconds)  # TTL 갱신
        except Exception as e:
            logger.error(f"처리 완료 표시 실패: {e}", exc_info=True)
    
    def delete_logs(self, keys: List[str]):
        """로그 삭제 (FAQ 생성 후 정리용)
        
        Args:
            keys: 삭제할 Redis 키 리스트
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        
        try:
            if keys:
                redis_client.delete(*keys)
                logger.info(f"질문 로그 삭제: {len(keys)}개")
        except Exception as e:
            logger.error(f"질문 로그 삭제 실패: {e}", exc_info=True)

