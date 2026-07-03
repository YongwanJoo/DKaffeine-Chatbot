"""ChatLog 데이터베이스 모델 (Java 백엔드 기준)"""
from sqlalchemy import Column, BigInteger, String, DateTime, ForeignKey, Text, Integer, Index
from sqlalchemy.sql import func
from app.infrastructure.persistence.models.base import Base
import enum


class ChatLogStatus(str, enum.Enum):
    """채팅 로그 상태 (ERD 기준)"""
    SUCCESS = "SUCCESS"
    REQUERY = "REQUERY"
    GUARDRAIL = "GUARDRAIL"  # ERD에는 "GAURDRAIL"로 표기되어 있으나 Java 코드는 "GUARDRAIL" 사용
    ERROR = "ERROR"


class ChatLog(Base):
    """채팅 로그 엔티티 (Java 백엔드: com.dkaffein.chat.domain.entity.ChatLog와 일치)
    
    Java 백엔드 기준:
    - id: Long (PK, 컬럼명: chat_log_id, @GeneratedValue)
    - query: String (TEXT, NOT NULL)
    - response: String (TEXT, NULL)
    - status: ChatLogStatus (ENUM, NOT NULL, SUCCESS/REQUERY/GUARDRAIL/ERROR)
    - categoryId: Long (FK → category.category_id, NULL)
    - inputTime: LocalDateTime (NOT NULL)
    - outputTime: LocalDateTime (NULL)
    - latencyMs: Integer (NULL)
    - chatModelId: Long (NOT NULL)
    - inputTokens: Integer (NULL) - LLM 입력 토큰 수
    - outputTokens: Integer (NULL) - LLM 출력 토큰 수
    - BaseEntity 상속: createdAt, updatedAt, deletedAt
    """
    __tablename__ = "chat_log"
    
    # Java: @Id @GeneratedValue, 컬럼명: chat_log_id
    id = Column("chat_log_id", BigInteger, primary_key=True, autoincrement=True)
    
    # Java: query (TEXT, NOT NULL)
    query = Column("query", Text, nullable=False)
    
    # Java: response (TEXT, NULL)
    response = Column("response", Text, nullable=True)
    
    # Java: status (ChatLogStatus enum, SUCCESS/REQUERY/GUARDRAIL/ERROR)
    status = Column("status", String(20), nullable=False)  # VARCHAR로 저장 (Enum 값)
    
    # Java: categoryId (FK → category.category_id, NULL)
    # 배포 환경: Java 백엔드의 category 테이블이 존재함 (com.dkaffein.datasource.domain.entity.Category)
    # 외래키 제약조건: use_alter=True로 나중에 추가 (category 테이블이 먼저 생성되어야 함)
    category_id = Column(
        "category_id", 
        BigInteger, 
        ForeignKey("category.category_id", name="fk_chat_log_category", use_alter=True), 
        nullable=True
    )
    
    # Java: chatModelId (NOT NULL)
    # 배포 환경: Java 백엔드의 chat_model 테이블이 존재함 (com.dkaffein.chatbotSetting.domain.entity.ChatModel)
    # 외래키 제약조건: 배포 환경에서는 필수, 테스트 환경에서는 init_db()에서 에러 무시
    # 주의: chat_model_id는 필수 필드이므로 항상 유효한 값이 들어가야 함 (기본값: 1)
    chat_model_id = Column(
        "chat_model_id", 
        BigInteger, 
        ForeignKey("chat_model.chat_model_id", name="fk_chat_log_chat_model"), 
        nullable=False
    )
    
    # Java: inputTime (NOT NULL)
    input_time = Column("input_time", DateTime(timezone=False), nullable=False)
    
    # Java: outputTime (NULL)
    output_time = Column("output_time", DateTime(timezone=False), nullable=True)
    
    # Java: latencyMs (NULL)
    latency_ms = Column("latency_ms", Integer, nullable=True)
    
    # Token usage (추가됨 - 비용 계산 및 통계용)
    prompt_tokens = Column("prompt_tokens", Integer, nullable=True)
    completion_tokens = Column("completion_tokens", Integer, nullable=True)
    
    # Java BaseEntity: createdAt, updatedAt, deletedAt
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)
    
    # Stability Fix: DB 인덱스 추가 (성능 최적화)
    # session_id는 ChatLog 모델에 없지만, 향후 추가될 수 있으므로 주석으로 남김
    # created_at 인덱스: 오래된 로그 조회 성능 향상
    __table_args__ = (
        Index('idx_chat_log_created_at', 'created_at'),
        # 향후 session_id 컬럼이 추가되면 아래 인덱스 활성화
        # Index('idx_chat_log_session_id', 'session_id'),
    )
    
    # 하위 호환성을 위한 chat_log_id 속성 (id와 동일)
    @property
    def chat_log_id(self):
        """하위 호환성: id를 chat_log_id로도 접근 가능"""
        return self.id

    def __repr__(self):
        return f"<ChatLog(id={self.id}, status={self.status})>"
