"""ChatModelConfig 데이터베이스 모델 (Java 백엔드 기준)"""
from sqlalchemy import Column, BigInteger, String, Integer, Text, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.infrastructure.persistence.models.base import Base


class ChatModelConfig(Base):
    """챗봇 모델 설정 엔티티 (Java 백엔드: com.dkaffein.chatbotSetting.domain.entity.ChatModelConfig와 일치)
    
    Java 백엔드 기준:
    - id: Long (PK, 컬럼명: model_config_id, @GeneratedValue)
    - chatModel: ChatModel (FK → chat_model.chat_model_id, NOT NULL)
    - temperature: Double (NOT NULL)
    - topP: Double (NOT NULL)
    - maxTokens: Integer (NOT NULL)
    - personaType: String (NULL)
    - systemPrompt: String (TEXT, NULL)
    - responseLength: String (NULL, short|medium|long)
    - searchResultCount: Integer (NOT NULL, RAG 검색 결과 수 3~7개 권장)
    - BaseEntity 상속: createdAt, updatedAt, deletedAt
    """
    __tablename__ = "chat_model_config"

    # Java: @Id @GeneratedValue, 컬럼명: model_config_id
    id = Column("model_config_id", BigInteger, primary_key=True, autoincrement=True)
    
    # Java: chatModel (FK → chat_model.chat_model_id, NOT NULL)
    chat_model_id = Column("chat_model_id", BigInteger, ForeignKey("chat_model.chat_model_id"), nullable=False)
    
    # Java: temperature (Double, NOT NULL)
    temperature = Column("temperature", Float, nullable=False)
    
    # Java: topP (Double, NOT NULL)
    top_p = Column("top_p", Float, nullable=False)
    
    # Java: maxTokens (Integer, NOT NULL)
    max_tokens = Column("max_tokens", Integer, nullable=False)
    
    # Java: personaType (String, NULL)
    persona_type = Column("persona_type", String(50), nullable=True)
    
    # Java: systemPrompt (TEXT, NULL)
    system_prompt = Column("system_prompt", Text, nullable=True)
    
    # Java: responseLength (String, NULL, short|medium|long)
    response_length = Column("response_length", String(50), nullable=True)
    
    # Java: searchResultCount (Integer, NOT NULL, RAG 검색 결과 수 3~7개 권장)
    search_result_count = Column("search_result_count", Integer, nullable=False)
    
    # Java BaseEntity: createdAt, updatedAt, deletedAt
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)
    
    # 하위 호환성을 위한 model_config_id 속성 (id와 동일)
    @property
    def model_config_id(self):
        """하위 호환성: id를 model_config_id로도 접근 가능"""
        return self.id

    def __repr__(self):
        return f"<ChatModelConfig(id={self.id}, chat_model_id={self.chat_model_id})>"
