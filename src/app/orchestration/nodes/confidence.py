"""Confidence 체크 노드"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from app.domain.constants import FAQ_MIN_CONFIDENCE
from app.domain.ports import CachePort
from .utils import _get_service_from_config

logger = logging.getLogger(__name__)


async def confidence_check_node(state: Union[ChatState, dict], config: dict) -> dict:
    """8-2단계: top_p 검증 및 답변 확정 (출처 포함)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    # config에서 서비스 가져오기
    cache_service: CachePort = _get_service_from_config(config, "cache_service")
    # FAQ 생성 서비스는 내부에서 직접 생성하므로 여기서는 가져오지 않음
    
    # RAG confidence threshold 설정 (설정에서 가져오기, 기본값: top_p 또는 0.8)
    chatbot_settings = chat_state.chatbot_settings
    thresholds = chatbot_settings.get("thresholds", {}) if chatbot_settings else {}
    rag_confidence_threshold = thresholds.get("rag_confidence_threshold", chat_state.top_p)
    
    confidence = chat_state.rag_confidence or 0.0
    rag_answer = chat_state.rag_answer or ""
    rag_sources = chat_state.rag_sources or []
    
    # 답변이 있고 출처가 있으면 confidence가 낮아도 통과 (실제로 답변이 생성되었으므로)
    # 단, confidence가 너무 낮으면 (0.3 미만) 거부
    has_valid_answer = bool(rag_answer and rag_answer.strip() and len(rag_answer.strip()) > 10)
    has_sources = bool(rag_sources and len(rag_sources) > 0)
    
    # Confidence 체크: 
    # 1. confidence >= 0.5: 통과
    # 2. 그 외: 실패 (Fallback으로 이동)
    
    # User Request: 신뢰도 0.5 미만은 무조건 실패 처리
    MIN_CONFIDENCE = 0.5
    
    confidence_passed = False
    confidence_reason = ""
    
    if confidence >= MIN_CONFIDENCE:
        confidence_passed = True
        confidence_reason = f"신뢰도 통과 (confidence={confidence:.3f} >= {MIN_CONFIDENCE})"
    else:
        confidence_passed = False
        confidence_reason = f"신뢰도 미달 (confidence={confidence:.3f} < {MIN_CONFIDENCE})"
    
    logger.info(f"confidence_check_node: {confidence_reason}, passed={confidence_passed}")
    
    if not confidence_passed:
        updated_state = chat_state.update(
            confidence_passed=False,
            should_generate_faq=False,  # 신뢰도 미달 시 FAQ 생성 후보에서 제외
            route=chat_state.route + " -> confidence_failed"
        )
        return to_state_dict(updated_state)
    
    # confidence_passed == True인 경우
    # 캐시 저장 (출처 정보 및 카테고리 포함) - 비동기 처리
    logger.info(f"confidence_check_node: 캐시 저장, rag_sources={rag_sources}, rag_category={chat_state.rag_category}")
    import asyncio
    await asyncio.to_thread(
        cache_service.set,
        chat_state.user_message,
        chat_state.rag_answer or "",
        sources=rag_sources,
        chatbot_settings=chatbot_settings,
        rag_category=chat_state.rag_category  # 카테고리 정보도 함께 저장
    )
    
    # 히스토리 업데이트: 현재 대화(질문+답변)를 히스토리에 추가
    current_history = list(chat_state.chat_history) if chat_state.chat_history else []
    
    # 현재 질문과 답변을 히스토리에 추가 (dict 형식으로 통일)
    final_answer = chat_state.rag_answer or ""
    current_history.append({"role": "user", "content": chat_state.user_message})
    current_history.append({"role": "assistant", "content": final_answer})
    
    # 최근 3턴(6개 메시지)만 유지 (메모리 최적화)
    if len(current_history) > 6:
        current_history = current_history[-6:]
    
    # FAQ 생성 여부 결정 (신뢰도 기반)
    # 높은 신뢰도의 RAG 답변은 FAQ 후보로 자동 생성
    should_generate_faq = confidence >= FAQ_MIN_CONFIDENCE
    
    updated_state = chat_state.update(
        confidence_passed=True,
        final_answer=final_answer,
        final_sources=rag_sources,
        chat_history=current_history,  # 업데이트된 히스토리 반환
        faq_count=chat_state.faq_count + 1,
        should_generate_faq=should_generate_faq,  # 백그라운드에서 FAQ 생성할지 여부
        route=chat_state.route + " -> confidence_passed"
    )
    return to_state_dict(updated_state)

