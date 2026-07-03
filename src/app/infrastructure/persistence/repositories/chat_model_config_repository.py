from typing import Optional, Tuple
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.persistence.models.chat_model_config_model import ChatModelConfig
from app.infrastructure.persistence.models.chat_model_model import ChatModel

class ChatModelConfigRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest_config_by_model_id(self, chat_model_id: int) -> Optional[ChatModelConfig]:
        """
        특정 챗봇 모델(chat_model_id)에 대한 최신 설정을 조회합니다.
        """
        # chat_model_id로 필터링하고, id(PK) 역순으로 정렬하여 최신 1개 조회
        stmt = (
            select(ChatModelConfig)
            .where(ChatModelConfig.chat_model_id == chat_model_id)
            .order_by(desc(ChatModelConfig.id))  # Java 백엔드와 일치: id 컬럼 사용
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_latest_config_with_model(self, chat_model_id: int) -> Optional[Tuple[ChatModelConfig, ChatModel]]:
        """
        특정 챗봇 모델(chat_model_id)에 대한 최신 설정과 모델 정보를 함께 조회합니다.
        
        Returns:
            (ChatModelConfig, ChatModel) 튜플 또는 None
        """
        # chat_model_config와 chat_model을 조인하여 조회
        stmt = (
            select(ChatModelConfig, ChatModel)
            .join(ChatModel, ChatModelConfig.chat_model_id == ChatModel.id)
            .where(ChatModelConfig.chat_model_id == chat_model_id)
            .order_by(desc(ChatModelConfig.id))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        row = result.first()
        if row:
            return (row[0], row[1])  # (ChatModelConfig, ChatModel)
        return None
