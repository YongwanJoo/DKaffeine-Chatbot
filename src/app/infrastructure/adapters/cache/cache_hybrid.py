"""하이브리드 캐시 (LRU + TTL)

FAQ 임베딩 캐시에 사용되는 하이브리드 캐시 구현입니다.
LRU (Least Recently Used)와 TTL (Time To Live)을 결합하여
메모리 사용량을 제한하면서도 자주 사용되는 항목을 유지합니다.
"""
import time
from collections import OrderedDict
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class HybridCache:
    """LRU + TTL 하이브리드 캐시
    
    두 가지 캐시 전략을 결합한 하이브리드 캐시:
    - **LRU (Least Recently Used)**: 자주 사용되는 항목을 우선적으로 유지
    - **TTL (Time To Live)**: 일정 시간이 지나면 자동으로 만료
    
    특징:
    - 메모리 사용량 제한 (max_size)
    - 자동 만료 (TTL)
    - 수동 무효화 지원
    - 캐시 통계 제공
    
    Example:
        ```python
        cache = HybridCache(max_size=1000, ttl_seconds=3600)
        
        # 저장
        cache.set("key1", "value1")
        
        # 조회
        value = cache.get("key1")
        
        # 통계
        stats = cache.get_stats()
        ```
    
    Args:
        max_size: 최대 캐시 크기 (LRU 제한, 기본값: 1000)
        ttl_seconds: TTL 초 (기본값: 3600, 1시간)
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            max_size: 최대 캐시 크기 (LRU 제한)
            ttl_seconds: TTL (초) - 기본 1시간
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        # OrderedDict: (key, (timestamp, value)) 형태
        self.cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        if key in self.cache:
            timestamp, value = self.cache[key]
            current_time = time.time()
            
            # TTL 체크
            if current_time - timestamp < self.ttl:
                # 최근 사용으로 이동 (LRU)
                self.cache.move_to_end(key)
                self._hits += 1
                return value
            else:
                # 만료된 항목 제거
                del self.cache[key]
                self._misses += 1
                logger.debug(f"Cache expired: {key}")
        else:
            self._misses += 1
        
        return None
    
    def set(self, key: str, value: Any):
        """캐시 저장"""
        current_time = time.time()
        
        if key in self.cache:
            # 기존 항목 업데이트 (최근 사용으로 이동)
            self.cache.move_to_end(key)
        else:
            # 새 항목 추가
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거 (LRU)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Cache evicted (LRU): {oldest_key}")
        
        self.cache[key] = (current_time, value)
    
    def invalidate(self, key: str):
        """특정 키 무효화 (FAQ 업데이트 시 사용)"""
        if key in self.cache:
            del self.cache[key]
            logger.info(f"Cache invalidated: {key}")
    
    def invalidate_pattern(self, pattern: str):
        """패턴으로 여러 키 무효화 (예: company_id별)"""
        keys_to_remove = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.cache[key]
        logger.info(f"Cache invalidated (pattern={pattern}): {len(keys_to_remove)} keys")
    
    def clear(self):
        """전체 캐시 초기화"""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """만료된 항목 일괄 정리"""
        current_time = time.time()
        expired_keys = [
            key for key, (timestamp, _) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl_seconds": self.ttl
        }

