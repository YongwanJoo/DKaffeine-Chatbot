from .state import ChatState
from typing import Literal, Union

def should_continue_after_blacklist(
    state: Union[ChatState, dict]
) -> Literal["unified_analysis", "__end__"]:
    """Blacklist 검증 후 분기"""
    if isinstance(state, dict):
        if state.get("blacklist_blocked", False):
            return "__end__"
    else:
        if state.blacklist_blocked:
            return "__end__"
    return "unified_analysis"

def should_continue_after_guardrail(
    state: Union[ChatState, dict]
) -> Literal["casual_response", "cache_check", "news_summary", "__end__"]:
    """통합 분석 후 분기 (일상 대화 vs 업무 질문 vs 뉴스 요약)"""
    if isinstance(state, dict):
        if not state.get("guardrail_passed", False):
            return "__end__"
        intent_type = state.get("intent_type")
    else:
        if not state.guardrail_passed:
            return "__end__"
        intent_type = state.intent_type
    
    if intent_type == "news":
        return "news_summary"
    if intent_type == "casual":
        return "casual_response"
    return "cache_check"


def should_use_cache(
    state: Union[ChatState, dict]
) -> Literal["faq_verify", "faq_search"]:
    """Cache 확인 후 분기"""
    if isinstance(state, dict):
        if state.get("cache_hit", False):
            return "faq_verify"
    else:
        if state.cache_hit:
            return "faq_verify"
    return "faq_search"

def should_use_rag(
    state: Union[ChatState, dict]
) -> Literal["rag", "__end__"]:
    """FAQ 처리 후 분기 (7-1, 7-2, 7-3 단계)"""
    # dict와 ChatState 모두 처리
    if isinstance(state, dict):
        faq_match = state.get("faq_match", False)
        answer_available = state.get("answer_available", False)
        chatbot_settings = state.get("chatbot_settings")
        faq_confidence = state.get("faq_confidence")
        cached_sources = state.get("cached_sources")
        final_sources = state.get("final_sources")
    else:
        faq_match = state.faq_match
        answer_available = state.answer_available
        chatbot_settings = state.chatbot_settings
        faq_confidence = state.faq_confidence
        cached_sources = state.cached_sources
        final_sources = state.final_sources
    
    # FAQ 매칭 또는 FAQ 확인 성공 시
    if faq_match or answer_available:
        import logging
        logger = logging.getLogger(__name__)
        
        # faq_confidence가 None이거나 0인 경우 (캐시 히트 등)
        if faq_confidence is None or faq_confidence == 0.0:
            # 캐시에서 온 경우 출처 확인
            # cached_sources 또는 final_sources 확인 (faq_verify_node에서 final_sources로 설정됨)
            sources = cached_sources or final_sources or []
            
            # sources가 비어있으면 캐시된 답변이 있지만 출처가 불명확 (RAG로 재시도)
            if not sources:
                logger.warning("Cache hit but sources are empty, proceeding to RAG")
                return "rag"
            
            # FAQ 출처인 경우 FAQ 답변으로 사용
            if sources == ["FAQ"] or (isinstance(sources, list) and len(sources) == 1 and sources[0] == "FAQ"):
                logger.info("Cache hit with FAQ sources, using cached FAQ answer")
                return "__end__"
            
            # FAQ가 아니고, sources가 존재하면(예: ["file.pdf"]) RAG 답변으로 간주
            # rag_service.py는 파일명만 저장하므로 s3:// 검사는 불필요
            if sources and len(sources) > 0:
                logger.info(f"Cache hit with RAG sources ({sources}), using cached RAG answer")
                return "__end__"
            
            # 그 외의 경우 (로직상 도달하기 어려움)
            logger.warning(f"Cache hit with unexpected sources format: {sources}, proceeding to RAG")
            return "rag"
        
        # FAQ confidence가 있는 경우: FAQ가 매칭되었다는 것은 이미 검색 임계값을 통과했다는 의미
        # 따라서 추가 confidence 체크 없이 바로 FAQ 답변 사용
        if faq_confidence is not None and faq_confidence > 0:
            logger.info(f"FAQ matched with confidence {faq_confidence:.3f}, using FAQ answer")
            return "__end__"
    # FAQ로 답변 불가능 → RAG로
    return "rag"

def should_route_after_rag(
    state: Union[ChatState, dict]
) -> Literal["confidence_check", "rag_fallback"]:
    """RAG 처리 후 분기: 답변이 있으면 confidence 체크, 없으면 fallback"""
    if isinstance(state, dict):
        if not state.get("has_answer", False):
            return "rag_fallback"
    else:
        if not state.has_answer:
            return "rag_fallback"
    return "confidence_check"

def should_finalize(
    state: Union[ChatState, dict]
) -> Literal["save_tokens", "rag_fallback"]:
    """Confidence 체크 후 분기"""
    if isinstance(state, dict):
        if state.get("confidence_passed", False):
            return "save_tokens"  # 9단계: 토큰 저장
    else:
        if state.confidence_passed:
            return "save_tokens"  # 9단계: 토큰 저장
    return "rag_fallback"  # Guardrail 처리
