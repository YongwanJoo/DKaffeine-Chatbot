"""유사도 검사 포트 인터페이스"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any


class SimilarityCheckerPort(ABC):
    """유사도 검사 포트 인터페이스
    
    FAQ 유사도 검사를 추상화하는 포트입니다.
    도메인 레이어는 이 인터페이스를 통해 유사도 검사에 접근합니다.
    """
    
    @abstractmethod
    def find_similar_question(
        self,
        question: str,
        company_id: Optional[str] = None,
        threshold: float = 0.8,
        limit: int = 5,
        question_embedding: Optional[List[float]] = None,
        session: Optional[Any] = None
    ) -> List[Any]:
        """유사한 질문 검색
        
        Args:
            question: 검색할 질문
            company_id: 회사 ID (선택적)
            threshold: 유사도 임계값
            limit: 최대 결과 수
            question_embedding: 질문 임베딩 (제공되면 재사용)
        
        Returns:
            유사한 FAQ 리스트
        """
        pass

