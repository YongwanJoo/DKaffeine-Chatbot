"""Blacklist 기반 Guardrail 노드"""
import logging
from typing import Union
from ..state import ChatState
from app.domain.ports import GuardrailPort
from .utils import _get_service_from_config, _get_detailed_block_reason

logger = logging.getLogger(__name__)


async def guardrail_blacklist_node(state: Union[ChatState, dict], config: dict) -> dict:
    """2-1단계: Blacklist 기반 1차 Guardrail (가장 빠른 검증)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    if isinstance(state, dict):
        chat_state = ChatState.from_dict(state)
    else:
        chat_state = state
    
    user_message = chat_state.user_message
    user_id = chat_state.user_id
    
    # config에서 서비스 가져오기 (포트 인터페이스)
    guardrail_service: GuardrailPort = _get_service_from_config(config, "guardrail_service")
    
    # 비동기 처리 (CPU 바운드지만 간단한 정규식 매칭이므로 바로 실행하거나 스레드풀 사용)
    # 여기서는 간단하므로 바로 실행 (필요시 asyncio.to_thread 사용)
    result = guardrail_service.check_blacklist(user_message)
    
    if result.blocked:
        logger.warning(
            f"Blacklist blocked: user={user_id}, "
            f"category={result.category}, "
            f"patterns={result.matched_patterns}"
        )
        
        # Blacklist 카테고리별 구체적인 메시지
        category = result.category or "general"
        detailed_reason = _get_detailed_block_reason(category, result.reason, result.details or {})
        
        # ChatState 업데이트 및 dict로 변환
        updated_state = chat_state.update(
            blacklist_blocked=True,
            blacklist_category=result.category,
            blacklist_reason=result.reason,
            blacklist_matched_patterns=result.matched_patterns,
            blocked=True,
            block_reason=detailed_reason,
            final_message=detailed_reason,
            route=chat_state.route + " -> blacklist_blocked"
        )
        return updated_state.to_dict()
    
    # ChatState 업데이트 및 dict로 변환
    updated_state = chat_state.update(
        blacklist_blocked=False,
        route=chat_state.route + " -> blacklist_passed"
    )
    return updated_state.to_dict()

