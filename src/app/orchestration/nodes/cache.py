"""Cache 확인 노드"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from app.domain.ports import CachePort
from .utils import _get_service_from_config

logger = logging.getLogger(__name__)


async def cache_check_node(state: Union[ChatState, dict], config: dict) -> dict:
    """4단계: Cache 확인 (Redis 메모리 조사)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    # config에서 서비스 가져오기
    cache_service: CachePort = _get_service_from_config(config, "cache_service")
    
    # Fix: 재질문(is_requery=True)인 경우 캐시 확인 스킵
    if getattr(chat_state, "is_requery", False):
        logger.info(f"cache_check_node: 재질문(is_requery=True)이므로 캐시 확인 스킵")
        updated_state = chat_state.update(
            cache_key=None,
            cache_hit=False,
            route=chat_state.route + " -> cache_skipped(requery)"
        )
        return to_state_dict(updated_state)
    
    # 비동기 처리 (Redis 호출)
    import asyncio
    chatbot_settings = chat_state.chatbot_settings
    
    cached_data = await asyncio.to_thread(
        cache_service.get_with_sources, 
        chat_state.user_message, 
        chatbot_settings=chatbot_settings
    )
    cache_key = await asyncio.to_thread(
        cache_service._make_key, 
        chat_state.user_message, 
        chatbot_settings=chatbot_settings
    )
    
    if cached_data:
        cached_answer = cached_data.get("answer")
        cached_sources = cached_data.get("sources")
        
        # Fix: cached_answer가 None이거나 빈 문자열인 경우 처리
        if not cached_answer or (isinstance(cached_answer, str) and not cached_answer.strip()):
            logger.warning(f"cache_check_node: 캐시 히트했지만 answer가 없음, cached_data={cached_data}")
            # answer가 없으면 캐시 미스로 처리
            updated_state = chat_state.update(
                cache_key=cache_key,
                cache_hit=False,
                route=chat_state.route + " -> cache_miss"
            )
            return to_state_dict(updated_state)
        
        logger.info(f"cache_check_node: 캐시 히트, cached_sources={cached_sources}, answer_len={len(cached_answer) if cached_answer else 0}")
        
        updated_state = chat_state.update(
            cache_key=cache_key,
            cache_hit=True,
            cached_response=cached_answer,
            cached_sources=cached_sources if cached_sources is not None else [],  # None이면 빈 리스트
            route=chat_state.route + " -> cache_hit",
            token_usage={"cache_hit_tokens": 500}  # 절약된 토큰
        )
        return to_state_dict(updated_state)
    
    updated_state = chat_state.update(
        cache_key=cache_key,
        cache_hit=False,
        route=chat_state.route + " -> cache_miss"
    )
    return to_state_dict(updated_state)

