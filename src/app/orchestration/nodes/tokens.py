"""토큰 사용량 저장 및 재실행 노드"""
import logging
from datetime import datetime
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict

logger = logging.getLogger(__name__)


async def save_tokens_node(state: Union[ChatState, dict], config: dict) -> dict:
    """9단계: 토큰 사용량 저장
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    token_usage = chat_state.token_usage or {}
    
    total_tokens = token_usage.get("total_tokens", 0)
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    cache_hit_tokens = token_usage.get("cache_hit_tokens", 0)
    
    db_record = {
        "user_id": chat_state.user_id,
        "session_id": chat_state.session_id,
        "message_id": f"{chat_state.session_id}_{datetime.now().timestamp()}",
        "tokens": total_tokens,
        "timestamp": datetime.now().isoformat()
    }
    
    # 토큰 사용량 저장 (향후 DB 저장 구현 예정)
    logger.info(f"Token usage saved: {db_record}")
    
    updated_state = chat_state.update(
        route=chat_state.route + " -> tokens_saved"
    )
    return to_state_dict(updated_state)


async def rerun_node(state: Union[ChatState, dict], config: dict) -> dict:
    """6단계: 재질문 처리
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    # 재질문 플래그가 설정되어 있으면 이전 질문을 히스토리에 추가하고 재시작
    if chat_state.needs_rerun:
        previous_q = list(chat_state.previous_questions)
        previous_q.append(chat_state.user_message)
        
        # 이전 대화를 chat_history에 추가
        chat_history = list(chat_state.chat_history) if chat_state.chat_history else []
        
        # 메시지 형식으로 추가
        chat_history.append({
            "role": "user",
            "content": chat_state.user_message
        })
        
        updated_state = chat_state.update(
            previous_questions=previous_q,
            chat_history=chat_history,
            needs_rerun=False,
            route=chat_state.route + " -> rerun_prepared"
        )
        return to_state_dict(updated_state)
    
    updated_state = chat_state.update(
        route=chat_state.route + " -> no_rerun"
    )
    return to_state_dict(updated_state)

