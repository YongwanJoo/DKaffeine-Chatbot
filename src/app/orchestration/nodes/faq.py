"""FAQ 검색 및 확인 노드"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from app.domain.ports import FAQPort, CachePort
from .utils import _get_service_from_config

logger = logging.getLogger(__name__)


async def faq_search_node(state: Union[ChatState, dict], config: dict) -> dict:
    """3-1단계: FAQ 검색 (Cache miss)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    chat_state = ensure_chat_state(state)
    
    faq_service: FAQPort = _get_service_from_config(config, "faq_service")
    cache_service: CachePort = _get_service_from_config(config, "cache_service")
    provider = _get_service_from_config(config, "provider")
    
    import asyncio
    
    user_embedding = chat_state.user_embedding
    if user_embedding is None:
        try:
            # 임베딩 생성 (비동기 처리)
            embeddings = provider.get_embeddings()
            user_embedding = await asyncio.to_thread(embeddings.embed_query, chat_state.user_message)
        except Exception as e:
            logger.warning(f"사용자 질문 임베딩 생성 실패: {e}")
            user_embedding = None
    
    chatbot_settings = chat_state.chatbot_settings
    thresholds = chatbot_settings.get("thresholds", {}) if chatbot_settings else {}
    # FAQ 검색 임계값을 낮춰서 더 관대하게 매칭 (0.70 → 0.60)
    faq_search_threshold = thresholds.get("faq_search_threshold", 0.60)
    
    # FAQ 검색 (비동기 처리)
    faq_match, faq_answer, faq_confidence = await asyncio.to_thread(
        faq_service.search,
        chat_state.user_message,
        threshold=faq_search_threshold,
        user_embedding=user_embedding
    )
    
    if faq_match:
        # 캐시 저장 (비동기 처리)
        await asyncio.to_thread(
            cache_service.set,
            chat_state.user_message,
            faq_answer,
            sources=["FAQ"],
            chatbot_settings=chatbot_settings
        )
        
        updated_state = chat_state.update(
            faq_match=True,
            faq_answer=faq_answer,
            faq_confidence=faq_confidence,
            final_answer=faq_answer,
            final_sources=["FAQ"],
            faq_count=chat_state.faq_count + 1,
            user_embedding=user_embedding,
            route=chat_state.route + " -> faq_matched"
        )
        return to_state_dict(updated_state)
    
    try:
        # 질문 로깅 (비동기 처리)
        from app.infrastructure.adapters.cache.question_log_service import QuestionLogService
        
        cache_redis_client = None
        if cache_service.use_redis and hasattr(cache_service, 'redis_client'):
            cache_redis_client = cache_service.redis_client
        
        log_service = QuestionLogService(redis_client=cache_redis_client)
        await asyncio.to_thread(
            log_service.log_question,
            question=chat_state.user_message,
            company_id="default",  # company_id 제거: 기본값 사용
            user_id=chat_state.user_id
        )
    except Exception as e:
        logger.debug(f"질문 로깅 실패 (무시): {e}")
    
    updated_state = chat_state.update(
        faq_match=False,
        user_embedding=user_embedding,
        route=chat_state.route + " -> faq_miss"
    )
    return to_state_dict(updated_state)


async def faq_verify_node(state: Union[ChatState, dict], config: dict) -> dict:
    """3-2단계: FAQ 확인 (Cache hit)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    
    Fix: cached_response가 없거나 빈 문자열인 경우 캐시에서 다시 조회
    """
    chat_state = ensure_chat_state(state)
    
    cache_service: CachePort = _get_service_from_config(config, "cache_service")
    
    # Fix: cached_response가 없거나 빈 문자열이면 캐시에서 다시 조회
    cached_response = chat_state.cached_response
    cached_sources = chat_state.cached_sources
    cached_category = None  # 캐시에서 복원할 카테고리 정보
    
    if not cached_response or (isinstance(cached_response, str) and not cached_response.strip()):
        import asyncio
        chatbot_settings = chat_state.chatbot_settings
        cached_data = await asyncio.to_thread(
            cache_service.get_with_sources, 
            chat_state.user_message,
            chatbot_settings=chatbot_settings
        )
        if cached_data:
            cached_response = cached_data.get("answer")
            if cached_sources is None:
                cached_sources = cached_data.get("sources")
            # 캐시에서 카테고리 정보도 복원
            cached_category = cached_data.get("rag_category")
        else:
            # 캐시에서도 찾을 수 없으면 실패
            logger.warning(f"faq_verify_node: 캐시 히트였지만 cached_response가 없고 재조회도 실패")
            updated_state = chat_state.update(
                answer_available=False,
                route=chat_state.route + " -> faq_verify_failed"
            )
            return to_state_dict(updated_state)
    
    # cached_response가 유효한 경우
    if cached_response and (not isinstance(cached_response, str) or cached_response.strip()):
        if cached_sources is None or cached_category is None:
            import asyncio
            chatbot_settings = chat_state.chatbot_settings
            cached_data = await asyncio.to_thread(
                cache_service.get_with_sources, 
                chat_state.user_message,
                chatbot_settings=chatbot_settings
            )
            if cached_data:
                if cached_sources is None:
                    cached_sources = cached_data.get("sources")
                if cached_category is None:
                    cached_category = cached_data.get("rag_category")
        
        # FAQ 히트 카운트 증가 로직 (RAG 캐시인 경우만)
        # 이미 FAQ인 경우(sources=["FAQ"])는 중복 카운트 방지를 위해 제외
        # RAG 답변이 캐시된 경우(sources=["file.pdf"] 등)는 빈도를 증가시켜 FAQ 후보로 등록 유도
        is_faq_source = False
        if cached_sources:
            is_faq_source = any("FAQ" in str(s) for s in cached_sources)
        
        if not is_faq_source:
            try:
                import asyncio
                from app.infrastructure.adapters.cache.question_log_service import QuestionLogService
                
                # Redis 클라이언트 가져오기
                cache_redis_client = None
                if cache_service.use_redis and hasattr(cache_service, 'redis_client'):
                    cache_redis_client = cache_service.redis_client
                
                log_service = QuestionLogService(redis_client=cache_redis_client)
                
                # 비동기로 로그 기록 (빈도 증가)
                await asyncio.to_thread(
                    log_service.log_question,
                    question=chat_state.user_message,
                    company_id="default",
                    user_id=chat_state.user_id
                )
                logger.info(f"faq_verify_node: RAG 캐시 히트, 질문 빈도 증가 (question='{chat_state.user_message}')")
            except Exception as e:
                logger.warning(f"faq_verify_node: 질문 빈도 증가 실패 (무시): {e}")
        
        # FAQ 생성 후보로 등록 (RAG 캐시인 경우)
        should_generate_faq = not is_faq_source
        
        # 캐시에서 복원한 카테고리 정보 사용 (없으면 기존 값 유지)
        rag_category = cached_category if cached_category is not None else chat_state.rag_category
        
        updated_state = chat_state.update(
            answer_available=True,
            final_answer=cached_response,
            final_sources=cached_sources if cached_sources is not None else [],
            cached_sources=cached_sources if cached_sources is not None else [],
            faq_count=chat_state.faq_count + 1,
            # FAQ 생성을 위한 필드 설정
            should_generate_faq=should_generate_faq,
            rag_answer=cached_response if should_generate_faq else None,
            rag_confidence=1.0 if should_generate_faq else None,  # 캐시 히트는 신뢰도 1.0 간주
            rag_category=rag_category,  # 캐시에서 복원한 카테고리 정보
            route=chat_state.route + " -> faq_verified"
        )
        logger.info(f"faq_verify_node: 캐시 답변 사용, answer_len={len(cached_response) if cached_response else 0}, sources={cached_sources}")
        return to_state_dict(updated_state)
    
    # cached_response가 유효하지 않은 경우
    logger.warning(f"faq_verify_node: cached_response가 유효하지 않음: {cached_response}")
    updated_state = chat_state.update(
        answer_available=False,
        route=chat_state.route + " -> faq_verify_failed"
    )
    return to_state_dict(updated_state)

