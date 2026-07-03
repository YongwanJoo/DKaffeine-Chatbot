"""질문 로그 어댑터"""
from typing import Optional, List

from app.domain.ports.question_log_port import QuestionLogPort
from app.infrastructure.adapters.cache.question_log_service import QuestionLogService


class QuestionLogAdapter(QuestionLogPort):
    """질문 로그 어댑터 구현
    
    QuestionLogPort 인터페이스를 구현하여 질문 로그 접근을 제공합니다.
    """
    
    def __init__(self, question_log_service: Optional[QuestionLogService] = None):
        """질문 로그 어댑터 초기화
        
        Args:
            question_log_service: QuestionLogService 인스턴스 (None이면 새로 생성)
        """
        self._service = question_log_service or QuestionLogService()
    
    def get_cluster_frequency(
        self,
        question: str,
        question_embedding: Optional[List[float]] = None,
        company_id: str = "default",
        similarity_threshold: float = 0.8,
        days_back: int = 30
    ) -> int:
        """의미 기반 클러스터 빈도수 조회"""
        return self._service.get_cluster_frequency(
            question=question,
            question_embedding=question_embedding,
            company_id=company_id,
            similarity_threshold=similarity_threshold,
            days_back=days_back
        )

