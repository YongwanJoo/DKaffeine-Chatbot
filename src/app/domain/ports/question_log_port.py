"""질문 로그 포트 인터페이스"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class QuestionLogPort(ABC):
    """질문 로그 포트 인터페이스
    
    질문 로그 및 빈도수 관리를 추상화하는 포트입니다.
    도메인 레이어는 이 인터페이스를 통해 질문 로그에 접근합니다.
    """
    
    @abstractmethod
    def get_cluster_frequency(
        self,
        question: str,
        question_embedding: Optional[List[float]] = None,
        company_id: str = "default",
        similarity_threshold: float = 0.8,
        days_back: int = 30
    ) -> int:
        """의미 기반 클러스터 빈도수 조회
        
        Args:
            question: 질문
            question_embedding: 질문 임베딩 (제공되면 재사용)
            company_id: 회사 ID
            similarity_threshold: 유사도 임계값
            days_back: 조회 기간 (일)
        
        Returns:
            클러스터 빈도수
        """
        pass

    @abstractmethod
    def get_all_logs(
        self,
        company_id: str = "default",
        limit: int = 10000,
        min_frequency: int = 1,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """모든 질문 로그 조회
        
        Args:
            company_id: 회사 ID
            limit: 조회 개수 제한
            min_frequency: 최소 빈도수
            days_back: 조회 기간 (일)
            
        Returns:
            질문 로그 리스트
        """
        pass

    @abstractmethod
    def update_cluster(
        self,
        keys: List[str],
        cluster_id: int
    ) -> None:
        """클러스터 정보 업데이트
        
        Args:
            keys: 로그 키 리스트
            cluster_id: 클러스터 ID
        """
        pass

