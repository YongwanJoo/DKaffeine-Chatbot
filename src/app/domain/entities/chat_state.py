"""챗봇 워크플로우 상태 정의 (Pydantic BaseModel)

타입 안전성 강화: Pydantic 모델로 런타임 검증 및 Optional 최소화
"""
from typing import Annotated, Optional, Union
from operator import add
from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages


def merge_token_usage(left: dict, right: dict) -> dict:
    """토큰 사용량 병합 함수"""
    result = left.copy() if left else {}
    if right:
        for key, value in right.items():
            if key in result:
                if isinstance(result[key], (int, float)) and isinstance(value, (int, float)):
                    result[key] = result[key] + value
                else:
                    result[key] = value
            else:
                result[key] = value
    return result


class ChatState(BaseModel):
    """챗봇 워크플로우 상태 (Pydantic BaseModel)
    
    LangGraph 호환성을 위해 dict로 변환 가능하며,
    런타임 타입 검증을 제공합니다.
    """
    model_config = {
        "extra": "allow",  # LangGraph가 추가 필드를 사용할 수 있도록 허용
        "validate_assignment": True,  # 할당 시 검증
    }
    
    # 입력 필드 (필수)
    user_message: str
    user_id: str
    session_id: str
    chat_history: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="대화 히스토리 (누적)"
    )
    
    # 2-1단계: Blacklist Guardrail
    blacklist_blocked: bool = False
    blacklist_category: Optional[str] = None
    blacklist_reason: Optional[str] = None
    blacklist_matched_patterns: Optional[list[str]] = None
    
    # 2-2단계: LLM Guardrail
    guardrail_passed: bool = False
    guardrail_reason: Optional[str] = None
    guardrail_category: Optional[str] = None
    guardrail_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    guardrail_check_details: Optional[dict] = None
    
    # 3단계: 의도 분석
    intent_type: Optional[str] = Field(None, description='"casual", "business", 또는 "news"')
    intent_category: Optional[str] = Field(
        None,
        description='"leave", "birth", "welfare", "dress_code", "general" 등'
    )
    intent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    intent_analysis_details: Optional[dict] = None
    
    # 4단계: Cache
    cache_key: Optional[str] = None
    cache_hit: bool = False
    cached_response: Optional[str] = None
    cached_sources: Optional[list[str]] = None
    
    # 4-1단계: FAQ 검색
    faq_match: bool = False
    faq_answer: Optional[str] = None
    faq_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_embedding: Optional[list[float]] = Field(
        None,
        description="사용자 질문 임베딩 (중복 생성 방지)"
    )
    
    # 4-2단계: FAQ 확인 (Cache hit)
    answer_available: bool = False
    
    # 5단계: RAG
    rag_answer: Optional[str] = None
    rag_sources: Optional[list[str]] = None
    rag_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    has_answer: bool = False
    related_queries: list[str] = Field(
        default_factory=list,
        description="연관 검색어 리스트 (relevance_score >= 0.7인 문서들에서 추출)"
    )
    
    # 6단계: top_p 검증
    confidence_passed: bool = False
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="신뢰도 임계값")
    
    # 챗봇 설정 (관리자 설정)
    chatbot_settings: Optional[dict] = Field(
        None,
        description="ChatbotSettings를 dict로 변환"
    )
    
    # 모델 정보
    model_used: Optional[str] = Field(None, description='"claude-sonnet-4-5" 등')
    
    # 7단계: 재질문
    needs_rerun: bool = False
    is_requery: bool = Field(default=False, description="재질문 여부 (캐시 무시)")
    previous_questions: list[str] = Field(default_factory=list)
    
    # 8단계: 토큰 사용량 (누적)
    rag_category: Optional[str] = None  # RAG 검색 결과의 주 카테고리
    token_usage: Annotated[dict, merge_token_usage] = Field(default_factory=dict)
    
    # 최종 출력
    final_answer: Optional[str] = None
    final_sources: Optional[list[str]] = None
    final_message: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    
    # 지표
    faq_count: int = Field(default=0, description="FAQ 카운트 (누적)")
    route: str = Field(default="", description="실행 경로 추적")
    
    # FAQ 생성
    faq_generated: Optional[bool] = Field(None, description="FAQ 후보 생성 여부")
    faq_generation_message: Optional[str] = Field(None, description="FAQ 생성 메시지")
    should_generate_faq: bool = False
    
    def to_dict(self) -> dict:
        """dict로 변환 (LangGraph 호환성)"""
        return self.model_dump(exclude_none=False, mode="python")
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChatState":
        """dict에서 ChatState 생성"""
        return cls(**data)
    
    def update(self, **kwargs) -> "ChatState":
        """상태 업데이트 (새 인스턴스 반환)"""
        return self.model_copy(update=kwargs)
    
    # LangGraph 호환성을 위한 dict 인터페이스
    def __getitem__(self, key: str):
        """dict처럼 접근 가능하도록 지원"""
        return getattr(self, key)
    
    def get(self, key: str, default=None):
        """dict처럼 get 메서드 지원"""
        return getattr(self, key, default)


# 헬퍼 함수: dict와 ChatState 간 변환
def ensure_chat_state(state: Union[ChatState, dict]) -> ChatState:
    """dict를 ChatState로 변환 (타입 검증)
    
    Args:
        state: ChatState 인스턴스 또는 dict
        
    Returns:
        ChatState 인스턴스
    """
    if isinstance(state, ChatState):
        return state
    return ChatState.from_dict(state)


def to_state_dict(state: Union[ChatState, dict]) -> dict:
    """ChatState를 dict로 변환 (LangGraph 호환성)
    
    Args:
        state: ChatState 인스턴스 또는 dict
        
    Returns:
        dict (LangGraph 호환)
    """
    if isinstance(state, dict):
        return state
    return state.to_dict()
