"""챗봇 설정 데이터베이스 모델"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, BigInteger, String, Text, Float, Integer, Boolean, DateTime, Enum as SQLEnum, JSON
from sqlalchemy.sql import func

from app.infrastructure.persistence.models.base import Base


class PersonaType(str, Enum):
    """페르소나 타입"""
    PROFESSIONAL = "professional"  # 전문적 (업무 지향적)
    FRIENDLY = "friendly"  # 친근한
    FORMAL = "formal"  # 격식있는
    CASUAL = "casual"  # 캐주얼


class ResponseLength(str, Enum):
    """응답 길이"""
    SHORT = "short"  # 짧음 (50-100자)
    NORMAL = "normal"  # 보통 (100-300자)
    LONG = "long"  # 김 (300-500자)
    VERY_LONG = "very_long"  # 매우 김 (500자 이상)


class LLMModel(str, Enum):
    """LLM 모델 타입
    
    주의: PostgreSQL enum은 짧은 이름만 지원하므로, 
    실제 모델 ID는 get_model_id() 함수를 통해 가져옵니다.
    """
    CLAUDE_SONNET_4_5 = "CLAUDE_SONNET_4_5"
    CLAUDE_3_5_SONNET = "CLAUDE_3_5_SONNET"
    CLAUDE_3_OPUS = "CLAUDE_3_OPUS"
    
    @staticmethod
    def get_model_id(model_enum: "LLMModel") -> str:
        """enum 값을 실제 모델 ID로 변환"""
        model_mapping = {
            LLMModel.CLAUDE_SONNET_4_5: "anthropic.claude-sonnet-4-5-20250929-v1:0",
            LLMModel.CLAUDE_3_5_SONNET: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            LLMModel.CLAUDE_3_OPUS: "anthropic.claude-3-opus-20240229-v1:0",
        }
        return model_mapping.get(model_enum, "anthropic.claude-sonnet-4-5-20250929-v1:0")


class ChatbotConfig(Base):
    """챗봇 설정 엔티티 (로컬 설정, ERD에 없음)
    
    ⚠️ 주의: 이 모델은 ERD 명세서에 없는 Python 백엔드의 로컬 설정 테이블입니다.
    공유 PostgreSQL의 chat_model_config 테이블과는 별개입니다.
    향후 마이그레이션 시 chat_model_config로 통합될 수 있습니다.
    
    ERD 명세서의 챗봇 설정은 chat_model_config 테이블을 참고하세요.
    """
    __tablename__ = "chatbot_config"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    company_id = Column(String(100), nullable=False, unique=True, index=True)  # 회사별 설정 (ERD에 없음)
    
    # 모델 설정
    llm_model = Column(SQLEnum(LLMModel), nullable=False, default=LLMModel.CLAUDE_SONNET_4_5)
    temperature = Column(Float, nullable=False, default=0.7)
    max_tokens = Column(Integer, nullable=False, default=2000)
    top_p = Column(Float, nullable=False, default=0.9)
    search_results_count = Column(Integer, nullable=False, default=5)  # RAG 검색 결과 수
    
    # 페르소나 설정
    persona_type = Column(SQLEnum(PersonaType), nullable=False, default=PersonaType.PROFESSIONAL)
    persona_description = Column(Text, nullable=False)
    response_length = Column(SQLEnum(ResponseLength), nullable=False, default=ResponseLength.NORMAL)
    
    # Guardrail 키워드 설정 (JSON)
    guardrail_keywords = Column(JSON, nullable=True, default=None)
    # 예시 구조:
    # {
    #   "blocked_topic_keywords": ["정치", "종교", ...],
    #   "business_keywords": ["휴가", "연차", ...],
    #   "casual_keywords": ["날씨", "맛집", ...],
    #   "bot_initiated_casual_keywords": ["음식", "맛집", ...]
    # }
    
    # 임계값 설정 (JSON)
    thresholds = Column(JSON, nullable=True, default=None)
    # 예시 구조:
    # {
    #   "faq_min_confidence": 0.85,
    #   "faq_generation_min_confidence": 0.85,
    #   "rag_confidence_threshold": 0.7
    # }
    
    # 활성화 여부
    is_active = Column(Boolean, nullable=False, default=True)
    
    # 타임스탬프
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<ChatbotConfig(id={self.id}, company_id={self.company_id}, model={self.llm_model})>"

