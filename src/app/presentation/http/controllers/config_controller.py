from fastapi import APIRouter, status, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.domain.services.chatbot_config_sync_service import ChatbotConfigSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["config"])

class ConfigSyncRequest(BaseModel):
    chat_model_id: Optional[int] = None

@router.put("/sync", status_code=status.HTTP_200_OK)
async def sync_config(request: ConfigSyncRequest):
    """
    챗봇 설정 동기화 API
    
    Java 백엔드에서 설정이 변경되었을 때 호출하여
    Python 백엔드의 캐시를 무효화하고 최신 설정을 즉시 반영하도록 합니다.
    """
    try:
        ChatbotConfigSyncService.invalidate_cache(request.chat_model_id)
        logger.info(f"Config cache invalidated via API. Model ID: {request.chat_model_id or 'ALL'}")
        return {"message": "Config synchronized successfully"}
    except Exception as e:
        logger.error(f"Failed to sync config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync config: {str(e)}"
        )
