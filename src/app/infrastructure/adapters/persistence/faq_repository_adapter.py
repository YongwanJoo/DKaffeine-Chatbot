"""FAQ Repository 어댑터"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.domain.ports.repository_port import FAQRepositoryPort
from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
from app.infrastructure.persistence.models.faq_model import Faq, FaqStatus


class FAQRepositoryAdapter(FAQRepositoryPort):
    """FAQ Repository 어댑터 구현
    
    FAQRepositoryPort 인터페이스를 구현하여 FAQ 데이터 접근을 제공합니다.
    """
    
    def __init__(self, session: Session):
        self.session = session
        self._repo = FaqRepository(session)
    
    def _faq_to_dict(self, faq: Faq) -> dict:
        """Faq 모델을 dict로 변환 (Java 백엔드와 일치)
        
        Java 백엔드 기준 필드만 포함:
        - id, question, answer, status, deleted, created_at, updated_at
        
        주의: embedding은 PostgreSQL에 저장되지 않고 Redis에만 저장됩니다.
        """
        return {
            "id": faq.id,  # Java 백엔드와 일치: id 필드 사용
            "question": faq.question,
            "answer": faq.answer,
            "status": faq.status.value if isinstance(faq.status, FaqStatus) else faq.status,
            "deleted": faq.deleted,  # Java 백엔드와 일치: deleted 필드
            "created_at": faq.created_at.isoformat() if faq.created_at else None,
            "updated_at": faq.updated_at.isoformat() if faq.updated_at else None,
            # 하위 호환성을 위한 필드 (Java 백엔드에 없음)
            "faq_id": faq.id,  # 하위 호환성: id와 동일
        }
    
    def find_by_id(self, faq_id: int) -> Optional[dict]:
        """ID로 FAQ 조회"""
        faq = self._repo.find_by_id(faq_id)
        return self._faq_to_dict(faq) if faq else None
    
    def find_by_keywords(
        self,
        keywords: List[str],
        status: str = "ACTIVE",
        limit: int = 10
    ) -> List[dict]:
        """키워드로 FAQ 검색 (Java 백엔드와 일치)"""
        faq_status = FaqStatus[status] if isinstance(status, str) else status
        faqs = self._repo.find_by_keywords(keywords, faq_status, limit)
        return [self._faq_to_dict(faq) for faq in faqs]
    
    def find_all_active(self, limit: Optional[int] = None) -> List[dict]:
        """활성화된 모든 FAQ 조회 (Java 백엔드와 일치)"""
        faqs = self._repo.find_all_active(limit or 100)
        return [self._faq_to_dict(faq) for faq in faqs]
    
    def create(
        self,
        question: str,
        answer: str,
        status: str = "PENDING",
        embedding: Optional[List[float]] = None  # Java 백엔드에 없음: 무시됨 (Redis에만 저장)
    ) -> dict:
        """FAQ 생성 (Java 백엔드와 일치)
        
        주의: embedding 파라미터는 하위 호환성을 위해 유지하지만,
        PostgreSQL에는 저장되지 않고 Redis에만 저장됩니다.
        """
        faq_status = FaqStatus[status] if isinstance(status, str) else status
        # FaqRepository의 create_pending 사용
        faq = self._repo.create_pending(question, answer)
        # embedding은 PostgreSQL에 저장되지 않음 (Redis에만 저장)
        if faq_status != FaqStatus.PENDING:
            faq.status = faq_status
        self.session.flush()
        return self._faq_to_dict(faq)
    
    def update(self, faq_id: int, **kwargs) -> Optional[dict]:
        """FAQ 업데이트 (Java 백엔드와 일치)"""
        faq = self._repo.find_by_id(faq_id)
        if not faq:
            return None
        
        # 필드 업데이트 (Java 백엔드와 일치하는 필드만)
        if 'question' in kwargs:
            faq.question = kwargs['question']
        if 'answer' in kwargs:
            faq.answer = kwargs['answer']
        if 'status' in kwargs:
            status = kwargs['status']
            faq.status = FaqStatus[status] if isinstance(status, str) else status
        # embedding은 PostgreSQL에 저장되지 않음 (Redis에만 저장되므로 무시)
        # if 'embedding' in kwargs:
        #     faq.embedding = kwargs['embedding']
        
        self.session.flush()
        return self._faq_to_dict(faq)
    
    def create_pending(
        self,
        question: str,
        answer: str,
        embedding: Optional[List[float]] = None  # Java 백엔드에 없음: 무시됨 (Redis에만 저장)
    ) -> dict:
        """FAQ 후보 생성 (PENDING 상태) - Java 백엔드와 일치
        
        주의: embedding 파라미터는 하위 호환성을 위해 유지하지만,
        PostgreSQL에는 저장되지 않고 Redis에만 저장됩니다.
        """
        faq = self._repo.create_pending(question, answer)
        # embedding은 PostgreSQL에 저장되지 않음 (Redis에만 저장)
        self.session.flush()
        return self._faq_to_dict(faq)

