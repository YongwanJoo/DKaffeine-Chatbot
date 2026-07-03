from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.infrastructure.persistence.models import ChatLog

class ChatLogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, chat_log: ChatLog) -> ChatLog:
        self.session.add(chat_log)
        await self.session.commit()
        await self.session.refresh(chat_log)
        return chat_log

    async def find_latest_by_query(self, query: str, chat_model_id: int | None, time_limit: object) -> object:
        """
        주어진 쿼리와 모델 ID로 가장 최근의 채팅 로그를 찾습니다.
        time_limit 이후의 로그만 검색합니다.
        chat_model_id가 None이면 모델 ID를 무시하고 검색합니다.
        """
        conditions = [
            ChatLog.query == query,
            ChatLog.created_at >= time_limit
        ]
        
        if chat_model_id is not None:
            conditions.append(ChatLog.chat_model_id == chat_model_id)
            
        stmt = (
            select(ChatLog)
            .where(*conditions)
            .order_by(ChatLog.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(self, chat_log_id: int, status: str) -> None:
        """
        채팅 로그의 상태를 업데이트합니다.
        """
        chat_log = await self.session.get(ChatLog, chat_log_id)
        if chat_log:
            chat_log.status = status
            await self.session.commit()
