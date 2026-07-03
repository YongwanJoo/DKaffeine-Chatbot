"""FAQ Repository"""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.infrastructure.persistence.models.faq_model import Faq, FaqStatus


class FaqRepository:
    """FAQ 데이터베이스 접근 레이어"""

    def __init__(self, session: Session):
        self.session = session

    def find_by_keywords(
        self,
        keywords: List[str],
        status: Optional[List[FaqStatus]] = None,
        limit: int = 10
    ) -> List[Faq]:
        """키워드로 FAQ 검색 (Java 백엔드: searchByKeywordAndStatus와 유사)
        
        Args:
            keywords: 검색 키워드 리스트
            status: FAQ 상태 리스트 (기본: ACTIVE). None이면 ACTIVE만 검색.
            limit: 최대 결과 수
            
        Returns:
            매칭된 FAQ 리스트
        """
        query = self.session.query(Faq)
        
        # 상태 필터링 (리스트 지원)
        if status is None:
            status = [FaqStatus.ACTIVE]
        elif not isinstance(status, list):
            status = [status]
            
        query = query.filter(Faq.status.in_(status))

        # 키워드 매칭 (question 또는 answer에 포함)
        if keywords:
            keyword_filters = []
            for keyword in keywords:
                if len(keyword) > 1:  # 2글자 이상만 검색
                    keyword_filters.append(
                        or_(
                            Faq.question.ilike(f"%{keyword}%"),
                            Faq.answer.ilike(f"%{keyword}%")
                        )
                    )
            
            if keyword_filters:
                # OR 조건으로 키워드 중 하나라도 매칭되면 반환
                query = query.filter(or_(*keyword_filters))

        return query.limit(limit).all()

    def find_by_id(self, faq_id: int) -> Optional[Faq]:
        """ID로 FAQ 조회"""
        return self.session.query(Faq).filter(
            and_(
                Faq.id == faq_id,  # Java 백엔드와 일치: id 컬럼 사용
                Faq.status == FaqStatus.ACTIVE
            )
        ).first()

    def find_all_active(
        self,
        limit: int = 100
    ) -> List[Faq]:
        """활성화된 모든 FAQ 조회 (Java 백엔드: findByStatus와 유사)
        
        Args:
            limit: 최대 결과 수
            
        Returns:
            활성화된 FAQ 리스트
        """
        query = self.session.query(Faq).filter(
            Faq.status == FaqStatus.ACTIVE
        )
        return query.limit(limit).all()

    def create_pending(
        self,
        question: str,
        answer: str
    ) -> Faq:
        """FAQ 후보 생성 (PENDING 상태) - Java 백엔드: Faq.create()와 유사
        
        Args:
            question: 질문
            answer: 답변
            
        Returns:
            생성된 FAQ 객체
        """
        faq = Faq.create(
            question=question,
            answer=answer,
            status=FaqStatus.PENDING
        )
        self.session.add(faq)
        self.session.flush()  # ID 생성
        return faq

    def find_similar_question(
        self,
        question: str,
        threshold: float = 0.8,
        limit: int = 5
    ) -> List[Faq]:
        """유사한 질문이 이미 존재하는지 확인
        
        Args:
            question: 확인할 질문
            threshold: 유사도 임계값 (기본: 0.8)
            limit: 최대 결과 수
            
        Returns:
            유사한 FAQ 리스트
        """
        # 간단한 키워드 기반 유사도 체크
        # 실제로는 벡터 유사도나 더 정교한 방법을 사용할 수 있음
        keywords = [q.strip() for q in question.split() if len(q.strip()) > 1]
        
        if not keywords:
            return []
        
        query = self.session.query(Faq)
        
        # 키워드 매칭 필터
        keyword_filters = []
        for keyword in keywords[:5]:  # 최대 5개 키워드만 사용
            keyword_filters.append(
                or_(
                    Faq.question.ilike(f"%{keyword}%"),
                    Faq.answer.ilike(f"%{keyword}%")
                )
            )
        
        if keyword_filters:
            query = query.filter(or_(*keyword_filters))
        
        return query.limit(limit).all()

