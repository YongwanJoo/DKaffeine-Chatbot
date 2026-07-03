"""Orchestration 노드 모듈

각 노드는 단일 책임 원칙에 따라 분리되어 있습니다.
"""
from .blacklist import guardrail_blacklist_node
from .unified_analysis import unified_analysis_node
from .casual_response import casual_response_node
from .cache import cache_check_node
from .faq import faq_search_node, faq_verify_node
from .rag import rag_answer_node, rag_fallback_node
from .confidence import confidence_check_node
from .tokens import save_tokens_node, rerun_node
from .news import news_summary_node

# 하위 호환성을 위한 별칭 (deprecated)
def guardrail_llm_node(state, config):
    """DEPRECATED: unified_analysis_node를 사용하세요"""
    return unified_analysis_node(state, config)

def intent_analysis_node(state, config):
    """DEPRECATED: unified_analysis_node를 사용하세요"""
    # unified_analysis_node가 이미 실행되었다면 그 결과를 사용
    if state.get("intent_type"):
        return state
    # 그렇지 않으면 unified_analysis_node 호출
    return unified_analysis_node(state, config)

def business_intent_classify_node(state, config):
    """DEPRECATED: unified_analysis_node를 사용하세요"""
    # unified_analysis_node가 이미 실행되었다면 그 결과를 사용
    if state.get("intent_category"):
        return state
    # 그렇지 않으면 unified_analysis_node 호출
    return unified_analysis_node(state, config)

__all__ = [
    "guardrail_blacklist_node",
    "unified_analysis_node",
    "casual_response_node",
    "cache_check_node",
    "faq_search_node",
    "faq_verify_node",
    "rag_answer_node",
    "rag_fallback_node",
    "confidence_check_node",
    "save_tokens_node",
    "rerun_node",
    "news_summary_node",
    # Deprecated
    "guardrail_llm_node",
    "intent_analysis_node",
    "business_intent_classify_node",
]

