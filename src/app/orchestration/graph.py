from langgraph.graph import StateGraph, END
from .state import ChatState
from .nodes import (
    guardrail_blacklist_node,
    unified_analysis_node,
    casual_response_node,
    cache_check_node,
    faq_search_node,
    faq_verify_node,
    rag_answer_node,
    rag_fallback_node,
    confidence_check_node,
    rerun_node,
    save_tokens_node,
    news_summary_node,
)
from .edges import (
    should_continue_after_blacklist,
    should_continue_after_guardrail,
    should_use_cache,
    should_use_rag,
    should_route_after_rag,
    should_finalize,
)

def _wrap_node_with_config(node_func):
    """
    노드 함수를 래퍼로 감싸서 config를 전달
    
    LangGraph는 config를 노드 함수에 직접 전달하지 않으므로,
    래퍼 함수를 통해 config를 노드 함수에 전달합니다.
    
    클린 아키텍처 준수: config는 필수입니다.
    
    LangGraph는 config를 {"configurable": {...}} 형식으로 받을 수 있으므로,
    이를 처리합니다.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    async def wrapper(state, config=None):
        # config가 없으면 에러 발생 (클린 아키텍처: 의존성 주입 필수)
        if config is None:
            raise ValueError(
                "config가 제공되지 않았습니다. "
                "messages.py에서 app.state.services를 config로 전달해야 합니다. "
                "클린 아키텍처 원칙에 따라 모든 서비스는 의존성 주입을 통해 전달되어야 합니다."
            )
        
        # LangGraph는 config를 {"configurable": {...}} 형식으로 받을 수 있음
        # 하지만 우리는 직접 딕셔너리를 전달하므로, configurable이 있으면 그 안의 값을 사용
        if isinstance(config, dict) and "configurable" in config:
            actual_config = config.get("configurable", {})
        else:
            actual_config = config
        
        return await node_func(state, actual_config)
    return wrapper


def create_chatbot_graph():
    """챗봇 그래프 생성"""
    
    workflow = StateGraph(ChatState)
    
    # 노드 추가 (config를 전달할 수 있도록 래퍼로 감싸기)
    workflow.add_node("blacklist_check", _wrap_node_with_config(guardrail_blacklist_node))
    workflow.add_node("unified_analysis", _wrap_node_with_config(unified_analysis_node))
    workflow.add_node("casual_response", _wrap_node_with_config(casual_response_node))
    workflow.add_node("news_summary", _wrap_node_with_config(news_summary_node))
    workflow.add_node("cache_check", _wrap_node_with_config(cache_check_node))
    workflow.add_node("faq_search", _wrap_node_with_config(faq_search_node))
    workflow.add_node("faq_verify", _wrap_node_with_config(faq_verify_node))
    workflow.add_node("rag", _wrap_node_with_config(rag_answer_node))
    workflow.add_node("rag_fallback", _wrap_node_with_config(rag_fallback_node))
    workflow.add_node("confidence_check", _wrap_node_with_config(confidence_check_node))
    workflow.add_node("rerun_prepare", _wrap_node_with_config(rerun_node))
    workflow.add_node("save_tokens", _wrap_node_with_config(save_tokens_node))
    
    # 엣지 연결
    workflow.set_entry_point("blacklist_check")
    
    # 1. Blacklist → 통합 분석 (Unified Analysis) 또는 END
    workflow.add_conditional_edges(
        "blacklist_check",
        should_continue_after_blacklist,
        {
            "unified_analysis": "unified_analysis",
            "__end__": END
        }
    )
    
    # 2. 통합 분석 → 뉴스 요약 또는 일상 대화 응답 또는 Cache 확인 또는 END
    workflow.add_conditional_edges(
        "unified_analysis",
        should_continue_after_guardrail,
        {
            "news_summary": "news_summary",
            "casual_response": "casual_response",
            "cache_check": "cache_check",
            "__end__": END  # guardrail_passed가 False인 경우
        }
    )
    
    # 3-1. 뉴스 요약 → END
    workflow.add_edge("news_summary", END)
    
    # 3-2. 일상 대화 응답 → END
    workflow.add_edge("casual_response", END)
    
    # 5. Cache 확인 → FAQ Verify or FAQ Search
    workflow.add_conditional_edges(
        "cache_check",
        should_use_cache,
        {
            "faq_verify": "faq_verify",
            "faq_search": "faq_search"
        }
    )
    
    # 6-1. FAQ 검색 (Cache miss) → RAG or END
    workflow.add_conditional_edges(
        "faq_search",
        should_use_rag,
        {
            "rag": "rag",
            "__end__": END
        }
    )
    
    # 6-2. FAQ 확인 (Cache hit) → RAG or END
    workflow.add_conditional_edges(
        "faq_verify",
        should_use_rag,
        {
            "rag": "rag",
            "__end__": END
        }
    )
    
    # 7. RAG 처리 → Confidence 또는 Fallback
    workflow.add_conditional_edges(
        "rag",
        should_route_after_rag,
        {
            "confidence_check": "confidence_check",
            "rag_fallback": "rag_fallback"
        }
    )
    
    # 8. Confidence 체크 → Save or Fallback
    workflow.add_conditional_edges(
        "confidence_check",
        should_finalize,
        {
            "save_tokens": "save_tokens",
            "rag_fallback": "rag_fallback"
        }
    )
    
    # Fallback → END
    workflow.add_edge("rag_fallback", END)
    
    # Save Tokens → END
    workflow.add_edge("save_tokens", END)
    
    return workflow.compile()

