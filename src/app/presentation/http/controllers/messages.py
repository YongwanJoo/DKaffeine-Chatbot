"""메시지 API 컨트롤러

Polish: 비즈니스 로직을 UseCase로 분리하여 컨트롤러는 단순히 요청/응답 변환만 담당
"""
from fastapi import APIRouter, HTTPException, status, Request, Depends
import logging
from typing import Dict, Optional, Tuple

from app.presentation.http.models.message_models import ChatRequest, ChatResponse
from app.domain.ports import GuardrailPort
from app.domain.services.chat_usecase import ChatUsecase
from app.presentation.http.dependencies import (
    get_guardrail_service,
    get_all_services,
)

logger = logging.getLogger(__name__)

# Perf: 설정 조회 캐싱은 Service 레벨에서 처리됨

# 라우터 생성
router = APIRouter(prefix="/api/v1", tags=["messages"])



@router.post("/messages/web", response_model=ChatResponse)
async def create_message_web(
    request: ChatRequest,
    http_request: Request,
    guardrail_service: GuardrailPort = Depends(get_guardrail_service),
    services_config: dict = Depends(get_all_services),
):
    """메시지 생성 API (Web/BE용 - 동기 로깅, 마크다운 형식)
    
    - **sync_log=True**: 채팅 로그를 즉시 저장하고 ID를 반환합니다.
    - **response_format="markdown"**: 마크다운 형식으로 응답을 생성합니다.
    - DKaffeine-BE의 Requery 기능을 위해 필요합니다.
    """
    # 웹 클라이언트는 마크다운 형식 사용
    request.response_format = "markdown"
    return await _process_chat_request(request, guardrail_service, services_config, sync_log=True)


@router.post("/messages/bot", response_model=ChatResponse)
async def create_message_bot(
    request: ChatRequest,
    http_request: Request,
    guardrail_service: GuardrailPort = Depends(get_guardrail_service),
    services_config: dict = Depends(get_all_services),
):
    """메시지 생성 API (Bot용 - 비동기 로깅)
    
    - **sync_log=False**: 채팅 로그를 백그라운드(Celery)에서 저장합니다.
    - **Session Strategy**: Webhook 환경 특성상 session_id 관리가 어려우므로,
      요청에 session_id가 없으면 **user_id를 session_id로 사용**하여 문맥을 유지합니다.
    """
    # Bot 클라이언트는 session_id가 없으면 user_id를 사용하여 문맥 유지
    if not request.session_id and request.user_id:
        request.session_id = request.user_id
        
    return await _process_chat_request(request, guardrail_service, services_config, sync_log=False)


async def _process_chat_request(
    request: ChatRequest,
    guardrail_service: GuardrailPort,
    services_config: dict,
    sync_log: bool
) -> ChatResponse:
    """채팅 요청 처리 공통 로직"""
    try:
        logger.info(
            f"Chat request (sync_log={sync_log}): user={request.user_id}, "
            f"chat_model_id={request.chat_model_id}, is_requery={request.is_requery}, message={request.message[:50]}..."
        )
        
        # Polish: UseCase 생성 및 메시지 처리 (비즈니스 로직 캡슐화)
        chat_usecase = ChatUsecase(
            guardrail_service=guardrail_service,
            services_config=services_config
        )
        
        # UseCase에 메시지 처리 위임
        response_data = await chat_usecase.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            chat_model_id=request.chat_model_id,
            sync_log=sync_log,
            is_requery=request.is_requery,
            response_format=request.response_format
        )
        
        logger.info(f"Chat response: route={response_data.get('route', '')}, chat_log_id={response_data.get('chat_log_id')}")
        
        # 응답 데이터 검증 및 정규화 (Pydantic 모델로 변환)
        return ChatResponse(
            answer=response_data.get("final_answer", ""),
            sources=response_data.get("final_sources", []),
            intent_type=response_data.get("intent_type", "general"),
            session_id=response_data.get("session_id", ""),
            blocked=response_data.get("blocked", False),
            block_reason=response_data.get("block_reason"),
            token_usage=response_data.get("token_usage", {}),
            route=response_data.get("route", "unknown"),
            faq_count=response_data.get("faq_count", 0),
            cache_hit=response_data.get("cache_hit", False),
            model_used=response_data.get("model_used"),
            related_queries=response_data.get("related_queries"),
            chat_log_id=response_data.get("chat_log_id")
        )
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



