"""ChatModel 데이터베이스 모델 (Java 백엔드 기준)"""
from sqlalchemy import Column, BigInteger, String, Float, DateTime
from sqlalchemy.sql import func
from app.infrastructure.persistence.models.base import Base


class ChatModel(Base):
    """챗봇 모델 엔티티 (Java 백엔드: com.dkaffein.chatbotSetting.domain.entity.ChatModel와 일치)
    
    Java 백엔드 기준:
    - id: Long (PK, 컬럼명: chat_model_id, @GeneratedValue)
    - chatModelName: String (NOT NULL)
    - inputCost: Double (NOT NULL) - 입력 토큰당 비용
    - outputCost: Double (NOT NULL) - 출력 토큰당 비용
    - BaseEntity 상속: createdAt, updatedAt, deletedAt
    """
    __tablename__ = "chat_model"

    # Java: @Id @GeneratedValue, 컬럼명: chat_model_id
    id = Column("chat_model_id", BigInteger, primary_key=True, autoincrement=True)
    
    # Java: chatModelName (NOT NULL)
    chat_model_name = Column("chat_model_name", String(255), nullable=False)
    
    # Java: inputCost (NOT NULL) - 입력 토큰당 비용
    input_cost = Column("input_cost", Float, nullable=False)
    
    # Java: outputCost (NOT NULL) - 출력 토큰당 비용
    output_cost = Column("output_cost", Float, nullable=False)
    
    # Java BaseEntity: createdAt, updatedAt, deletedAt
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)

    def __repr__(self):
        return f"<ChatModel(id={self.id}, name={self.chat_model_name})>"

