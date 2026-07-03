"""메시지 API 요청/응답 모델"""
from pydantic import BaseModel, Field, validator, ConfigDict, model_validator
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """메시지 요청
    
    자바 백엔드에서 호출 시:
    - message: 필수 (사용자 질문)
    - user_id: 선택 (기본값: "user_001")
    - session_id: 선택 (없으면 자동 생성, 있으면 Redis에서 히스토리 자동 로드)
    - chat_model_id: 선택 (챗봇 모델 ID, 없으면 기본값 1 사용)
    
    참고:
    - chat_history: 제거됨 (session_id로 Redis에서 자동 로드, 보안상 클라이언트 제공 히스토리는 받지 않음)
    - company_id: 제거됨 (B2B 고려 안함, FAQ 생성 시 항상 "default" 사용)
    """
    model_config = ConfigDict(
        populate_by_name=True,  # alias(isRequery)와 필드명(is_requery) 모두 허용
        json_schema_extra={
            "example": {
                "message": "판교 주변 맛집 추천해줘",
                "user_id": "user_001",
                "session_id": "session_1234567890_abc123",
                "chat_model_id": 1,
                "isRequery": True
            }
        }
    )
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="사용자 질문 내용 (최대 5000자)",
        examples=["판교 주변 맛집 추천해줘", "카카오 워크에 대해 알려줘"]
    )
    user_id: str = Field(
        default="user_001",
        max_length=100,
        description="사용자 ID (기본값: user_001)",
        examples=["user_001", "user_002"]
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="세션 ID (없으면 자동 생성, 있으면 Redis에서 히스토리 자동 로드)",
        examples=["session_1234567890_abc123", None]
    )
    chat_model_id: Optional[int] = Field(
        default=None,
        description="챗봇 모델 ID (선택, 없으면 기본값 1 사용)",
        examples=[1, 2, None]
    )
    is_requery: bool = Field(
        default=False,
        description="재질문 여부 (True면 로그 상태를 REQUERY로 저장)",
        alias="isRequery",  # 프론트엔드에서 camelCase로 보내는 경우 지원
        examples=[False, True]
    )
    response_format: str = Field(
        default="plain",
        description="응답 형식 (plain: 일반 텍스트, markdown: 마크다운 형식)",
        examples=["plain", "markdown"]
    )
    
    @validator('message')
    def validate_message(cls, v):
        """메시지 검증"""
        if not v or not v.strip():
            raise ValueError('메시지는 비어있을 수 없습니다')
        return v.strip()
    
    @validator('response_format')
    def validate_response_format(cls, v):
        """응답 형식 검증"""
        if v not in ["plain", "markdown"]:
            raise ValueError('response_format은 "plain" 또는 "markdown"이어야 합니다')
        return v

    @model_validator(mode='before')
    @classmethod
    def log_raw_input(cls, data: Any) -> Any:
        """디버깅: 입력 데이터 로깅"""
        import logging
        logger = logging.getLogger("app.presentation.http.models.message_models")
        if isinstance(data, dict):
            is_requery_val = data.get('isRequery') or data.get('is_requery')
            logger.info(f"🔍 [ChatRequest Raw Input] keys={list(data.keys())}, isRequery={data.get('isRequery')}, is_requery={data.get('is_requery')}, resolved={is_requery_val}")
        return data


class ChatResponse(BaseModel):
    """메시지 응답
    
    챗봇의 응답 데이터를 포함합니다.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "판교 주변에는 다양한 맛집이 있습니다...",
                "sources": ["디케이테크인_회사소개서_2025_kor.pdf"],
                "blocked": False,
                "block_reason": None,
                "token_usage": {"input_tokens": 100, "output_tokens": 200},
                "route": "start -> blacklist_passed -> rag_executed",
                "faq_count": 0,
                "cache_hit": False,
                "session_id": "session_1234567890_abc123",
                "intent_type": "general",
                "intent_category": None,
                "model_used": "claude-haiku-4-5"
            }
        }
    )
    
    answer: Optional[str] = Field(
        default=None,
        description="챗봇의 답변 내용",
        examples=["판교 주변에는 다양한 맛집이 있습니다...", None]
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="답변에 사용된 문서 소스 목록",
        examples=[["디케이테크인_회사소개서_2025_kor.pdf"], []]
    )
    blocked: bool = Field(
        ...,
        description="차단 여부 (Guardrail에 의해 차단되었는지)",
        examples=[False, True]
    )
    block_reason: Optional[str] = Field(
        default=None,
        description="차단 사유 (blocked가 True일 때만 존재)",
        examples=[None, "부적절한 내용 감지"]
    )
    token_usage: Dict[str, Any] = Field(
        ...,
        description="토큰 사용량 (input_tokens, output_tokens 등)",
        examples=[{"input_tokens": 100, "output_tokens": 200}]
    )
    route: str = Field(
        ...,
        description="처리 경로 (LangGraph 워크플로우 경로)",
        examples=["start -> blacklist_passed -> rag_executed", "start -> guardrail_blocked"]
    )
    faq_count: int = Field(
        ...,
        description="매칭된 FAQ 개수",
        examples=[0, 1, 3]
    )
    cache_hit: bool = Field(
        ...,
        description="캐시 히트 여부",
        examples=[False, True]
    )
    session_id: str = Field(
        ...,
        description="세션 ID (멀티턴 대화를 위한 식별자)",
        examples=["session_1234567890_abc123"]
    )
    intent_type: Optional[str] = Field(
        default=None,
        description="의도 타입 (general, casual, business 등)",
        examples=["general", "casual", "business", None]
    )
    intent_category: Optional[str] = Field(
        default=None,
        description="의도 카테고리 (세부 분류)",
        examples=[None, "question", "greeting"]
    )
    model_used: Optional[str] = Field(
        default=None,
        description="사용된 LLM 모델 이름",
        examples=["claude-haiku-4-5", "claude-sonnet-4-5", None]
    )
    related_queries: Optional[List[str]] = Field(
        default=None,
        description="연관 검색어 리스트 (relevance_score >= 0.7인 문서들에서 추출)",
        examples=[["휴가 신청", "연차 사용", "휴가 취소"], None]
    )
    chat_log_id: Optional[int] = Field(
        default=None,
        description="채팅 로그 ID (동기 로깅 시 반환, 비동기 시 None)",
        examples=[12345, None]
    )



