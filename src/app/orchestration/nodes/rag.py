"""RAG 기반 답변 생성 노드"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from app.domain.ports import RAGPort
from .utils import _get_service_from_config

logger = logging.getLogger(__name__)


async def rag_answer_node(state: Union[ChatState, dict], config: dict) -> dict:
    """6단계: RAG 기반 답변 생성 (설정 기반)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    # config에서 서비스 가져오기
    rag_service: RAGPort = _get_service_from_config(config, "rag_service")
    
    # 업무 질문이면 Claude Sonet 사용 (설정에서 모델 선택)
    use_claude = chat_state.intent_type == "business"
    
    # 챗봇 설정 가져오기
    settings = chat_state.chatbot_settings
    
    # 응답 형식 가져오기 (config에서 전달됨)
    response_format = config.get("response_format", "plain")
    
    # 검색 결과 수 축소 (응답 속도 최적화)
    default_top_k = 4  # 기본값을 10에서 4로 축소 (응답 속도 향상)
    
    # 비동기 검색 실행
    if hasattr(rag_service, 'asearch'):
        has_answer, answer, sources, confidence, doc_number_to_filename, related_queries, primary_category = await rag_service.asearch(
            chat_state.user_message,
            chat_state.chat_history,
            top_k=settings.get("search_results_count", default_top_k) if settings else default_top_k,
            use_claude=use_claude,
            settings=settings,
            confidence_method="reranker",  # Reranker를 기본값으로 사용
            response_format=response_format  # 응답 형식 전달
        )
    else:
        # 동기 메서드 폴백 (스레드 풀 사용)
        import asyncio
        has_answer, answer, sources, confidence, doc_number_to_filename, related_queries, primary_category = await asyncio.to_thread(
            rag_service.search,
            chat_state.user_message,
            chat_state.chat_history,
            top_k=settings.get("search_results_count", default_top_k) if settings else default_top_k,
            use_claude=use_claude,
            settings=settings,
            confidence_method="reranker",
            response_format=response_format  # 응답 형식 전달
        )
    
    # Perf: LLM 모델 경량화 (claude-sonnet-4-5 -> claude-haiku-4-5)
    # 설정에서 모델을 주입받도록 변경 (기본값: claude-haiku-4-5)
    # Fix: settings 키는 'llm_model'임 ('model_name' 아님)
    model_used = settings.get("llm_model", "claude-haiku-4-5") if settings else "claude-haiku-4-5"
    
    logger.info(f"📂 [RAG] primary_category extracted: {primary_category}")
    
    updated_state = chat_state.update(
        has_answer=has_answer,
        rag_answer=answer,  # 원본 텍스트만 저장 (서식 처리 없음)
        rag_sources=sources,  # 원본 출처 리스트만 저장
        rag_confidence=confidence,
        model_used=model_used,
        route=chat_state.route + " -> rag_executed",
        token_usage={"prompt_tokens": 1500, "completion_tokens": 300, "total_tokens": 1800},
        related_queries=related_queries,  # 연관 검색어 추가
        should_generate_faq=has_answer,  # FAQ 생성 후보로 등록
        rag_category=primary_category  # RAG 카테고리 저장
    )
    return to_state_dict(updated_state)


async def rag_fallback_node(state: Union[ChatState, dict], config: dict) -> dict:
    """8-1단계: RAG 실패 시 Guardrail 처리
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    # final_message를 final_answer로도 설정 (format_response에서 사용)
    fallback_message = "죄송합니다. 제공된 문서에서 해당 질문에 대한 답변을 찾을 수 없습니다."
    updated_state = chat_state.update(
        final_message=fallback_message,
        final_answer=fallback_message,  # format_response에서 final_answer를 우선 사용하므로 설정
        route=chat_state.route + " -> rag_fallback"
    )
    return to_state_dict(updated_state)

