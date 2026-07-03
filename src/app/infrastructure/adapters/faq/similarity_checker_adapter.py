"""FAQ 유사도 검사 어댑터"""
from typing import Optional, List, Any
from sqlalchemy.orm import Session

from app.domain.ports.similarity_checker_port import SimilarityCheckerPort
from app.infrastructure.adapters.faq.faq_similarity_checker import FAQSimilarityChecker


class SimilarityCheckerAdapter(SimilarityCheckerPort):
    """FAQ 유사도 검사 어댑터 구현
    
    SimilarityCheckerPort 인터페이스를 구현하여 FAQ 유사도 검사를 제공합니다.
    """
    
    def __init__(self, session: Optional[Session] = None):
        """FAQ 유사도 검사 어댑터 초기화
        
        Args:
            session: SQLAlchemy 세션 (선택적)
        """
        self._checker = FAQSimilarityChecker(session=session)
        self._session = session
    
    def find_similar_question(
        self,
        question: str,
        company_id: Optional[str] = None,
        threshold: float = 0.8,
        limit: int = 5,
        question_embedding: Optional[List[float]] = None,
        session: Optional[Session] = None
    ) -> List[Any]:
        """유사한 질문 검색"""
        # 어댑터 초기화 시 받은 세션이 있으면 그것을 사용, 없으면 메서드 인자로 받은 세션 사용
        effective_session = session if session else self._session
        
        return self._checker.find_similar_question(
            question=question,
            company_id=company_id,
            threshold=threshold,
            limit=limit,
            session=effective_session,
            question_embedding=question_embedding
        )

