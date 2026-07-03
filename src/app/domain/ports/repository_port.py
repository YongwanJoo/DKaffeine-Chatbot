"""Repository 포트 인터페이스"""
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, TypeVar, Generic

# 타입 변수
T = TypeVar('T')


class RepositoryPort(ABC, Generic[T]):
    """Repository 포트 인터페이스
    
    데이터베이스 접근을 추상화하는 포트입니다.
    도메인 레이어는 이 인터페이스를 통해 데이터 접근을 수행합니다.
    """
    
    @abstractmethod
    def find_by_id(self, entity_id: int) -> Optional[T]:
        """ID로 엔티티 조회"""
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None) -> List[T]:
        """모든 엔티티 조회"""
        pass
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """엔티티 저장"""
        pass
    
    @abstractmethod
    def delete(self, entity_id: int) -> bool:
        """엔티티 삭제"""
        pass


class FAQRepositoryPort(ABC):
    """FAQ Repository 포트 인터페이스"""
    
    @abstractmethod
    def find_by_id(self, faq_id: int) -> Optional[dict]:
        """ID로 FAQ 조회"""
        pass
    
    @abstractmethod
    def find_by_keywords(
        self,
        keywords: List[str],
        status: str = "ACTIVE",
        limit: int = 10
    ) -> List[dict]:
        """키워드로 FAQ 검색 (Java 백엔드와 일치)"""
        pass
    
    @abstractmethod
    def find_all_active(self, limit: Optional[int] = None) -> List[dict]:
        """활성화된 모든 FAQ 조회 (Java 백엔드와 일치)"""
        pass
    
    @abstractmethod
    def create(
        self,
        question: str,
        answer: str,
        status: str = "PENDING",
        embedding: Optional[List[float]] = None  # Java 백엔드에 없음: 하위 호환성 (Redis에만 저장)
    ) -> dict:
        """FAQ 생성 (Java 백엔드와 일치)
        
        주의: embedding 파라미터는 하위 호환성을 위해 유지하지만,
        PostgreSQL에는 저장되지 않고 Redis에만 저장됩니다.
        """
        pass
    
    @abstractmethod
    def update(self, faq_id: int, **kwargs) -> Optional[dict]:
        """FAQ 업데이트"""
        pass
    
    @abstractmethod
    def create_pending(
        self,
        question: str,
        answer: str,
        embedding: Optional[List[float]] = None  # Java 백엔드에 없음: 하위 호환성 (Redis에만 저장)
    ) -> dict:
        """FAQ 후보 생성 (PENDING 상태) - Java 백엔드와 일치
        
        Args:
            question: 질문
            answer: 답변
            embedding: 임베딩 벡터 (선택적, PostgreSQL에는 저장되지 않고 Redis에만 저장)
        
        Returns:
            생성된 FAQ dict
        """
        pass


class ConfigRepositoryPort(ABC):
    """Config Repository 포트 인터페이스"""
    
    @abstractmethod
    def find_by_company_id(self, company_id: str) -> Optional[dict]:
        """회사 ID로 설정 조회"""
        pass
    
    @abstractmethod
    def find_default(self) -> Optional[dict]:
        """기본 설정 조회 (company_id='default')"""
        pass
    
    @abstractmethod
    def save(self, company_id: str, config: dict) -> dict:
        """설정 저장"""
        pass

