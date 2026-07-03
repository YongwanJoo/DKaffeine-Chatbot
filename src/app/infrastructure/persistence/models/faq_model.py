"""FAQ 데이터베이스 모델"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, BigInteger, Text, Boolean, DateTime, Enum as SQLEnum, Index
from sqlalchemy.sql import func

from app.infrastructure.persistence.models.base import Base


class FaqStatus(str, Enum):
    """FAQ 상태 (Java 백엔드 기준: 대문자)"""
    PENDING = "PENDING"  # AI 생성 등 승인 대기
    ACTIVE = "ACTIVE"    # 관리자 승인완료 혹은 수동 등록


class Faq(Base):
    """FAQ 엔티티 (Java 백엔드: com.dkaffein.faq.domain.entity.Faq와 일치)
    
    Java 백엔드 기준:
    - id: Long (PK, 컬럼명: id, @GeneratedValue)
    - question: String (TEXT, NOT NULL)
    - answer: String (TEXT, NOT NULL)
    - status: FaqStatus (ENUM, NOT NULL, PENDING/ACTIVE)
    - deleted: boolean (NOT NULL, 기본값: false)
    - BaseEntity 상속: createdAt, updatedAt, deletedAt
    
    주의: Java는 @Table(name = "faq")이지만 실제 컬럼명은 id입니다.
    Python에서는 하위 호환성을 위해 faq_id를 사용하되, id 속성으로 접근 가능합니다.
    """
    __tablename__ = "faq"

    # Java: @Id @GeneratedValue, 컬럼명: id
    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    
    # Java: question (TEXT, NOT NULL)
    question = Column(Text, nullable=False)
    
    # Java: answer (TEXT, NOT NULL)
    answer = Column(Text, nullable=False)
    
    # Java: status (FaqStatus enum, PENDING/ACTIVE)
    status = Column(SQLEnum(FaqStatus), nullable=False)
    
    # Java: deleted (boolean, NOT NULL, 기본값: false)
    deleted = Column(Boolean, nullable=False, default=False)
    
    # Java BaseEntity: createdAt, updatedAt, deletedAt
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)

    @staticmethod
    def create(question: str, answer: str, status: FaqStatus) -> 'Faq':
        """FAQ 생성 팩토리 메서드"""
        faq = Faq()
        faq.question = question
        faq.answer = answer
        faq.status = status
        
        # Fix: 타임스탬프 명시적 설정 (DB default에 의존하지 않음)
        now = datetime.now()
        faq.created_at = now
        faq.updated_at = now
        
        return faq

    def __repr__(self):
        return f"<Faq(id={self.id}, question={self.question[:50]}..., status={self.status}, deleted={self.deleted})>"


# 인덱스 생성
Index('ix_faq_status', Faq.status)

