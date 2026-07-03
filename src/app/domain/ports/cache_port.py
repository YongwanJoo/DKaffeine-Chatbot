"""Cache 서비스 포트 (인터페이스)"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class CachePort(ABC):
    """Cache 서비스 포트 (인터페이스)
    
    캐시 서비스의 추상 인터페이스입니다.
    모든 Cache 서비스 구현체는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def get(self, user_message: str) -> Optional[str]:
        """캐시 조회 (하위 호환성을 위해 문자열만 반환)
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            캐시된 답변 (없으면 None)
        """
        pass
    
    @abstractmethod
    def get_with_sources(self, user_message: str) -> Optional[Dict[str, Any]]:
        """캐시 조회 (답변과 출처 정보 포함)
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            캐시된 데이터 딕셔너리 (없으면 None)
            - answer: 답변 텍스트
            - sources: 출처 리스트
        """
        pass
    
    @abstractmethod
    def set(
        self,
        user_message: str,
        answer: str,
        sources: Optional[List[str]] = None,
        ttl: int = 3600
    ) -> None:
        """캐시 저장
        
        Args:
            user_message: 사용자 메시지
            answer: 답변 텍스트
            sources: 출처 리스트 (선택적)
            ttl: TTL (초, 기본값: 3600)
        """
        pass

    @abstractmethod
    def get_raw(self, key: str) -> Optional[str]:
        """Raw 키로 캐시 조회 (해싱 없음)
        
        Args:
            key: 캐시 키
            
        Returns:
            캐시된 값 (없으면 None)
        """
        pass
    
    @abstractmethod
    def set_raw(self, key: str, value: str, ttl: int = 3600) -> None:
        """Raw 키로 캐시 저장 (해싱 없음)
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초, 기본값: 3600)
        """
        pass
    
    def _make_key(self, user_message: str) -> str:
        """캐시 키 생성 (구현 세부사항, 선택적)
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            캐시 키 문자열
        """
        # 기본 구현 (하위 호환성)
        import hashlib
        normalized = user_message.strip().lower()
        hash_val = hashlib.sha256(normalized.encode()).hexdigest()
        return f"cache:v1.0:{hash_val}"

