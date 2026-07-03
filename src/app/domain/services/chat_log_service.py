from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.models import ChatLog, ChatLogStatus
from app.infrastructure.persistence.repositories.chat_log_repository import ChatLogRepository

class ChatLogService:
    def __init__(self, session: AsyncSession):
        self.repository = ChatLogRepository(session)

    async def save_log(
        self,
        query: str,
        response: str,
        status: str,
        chat_model_id: int,
        input_time: datetime,
        output_time: Optional[datetime] = None,
        latency_ms: Optional[int] = None,
        category_id: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        guardrail_reason: Optional[str] = None
    ) -> ChatLog:
        """
        채팅 로그를 저장합니다.
        """
        # Status Enum 변환 확인
        try:
            log_status = ChatLogStatus(status)
        except ValueError:
            log_status = ChatLogStatus.ERROR

        # created_at을 명시적으로 설정 (데이터베이스 default가 없을 수 있음)
        now = datetime.now()
        
        chat_log = ChatLog(
            query=query,
            response=response,
            status=log_status.value, # DB에는 String으로 저장
            chat_model_id=chat_model_id,
            input_time=input_time,
            output_time=output_time,
            latency_ms=latency_ms,
            category_id=category_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            created_at=now,  # 명시적으로 설정
            updated_at=now   # 명시적으로 설정
        )
        
        saved_log = await self.repository.create(chat_log)

        # 가드레일 차단인 경우 상세 내역 저장
        if log_status == ChatLogStatus.GUARDRAIL and guardrail_reason:
            from app.infrastructure.persistence.models import Guardrail
            guardrail_log = Guardrail(
                chat_log_id=saved_log.id,
                reason=guardrail_reason,
                created_at=now,
                updated_at=now
            )
            session = self.repository.session
            session.add(guardrail_log)
            await session.commit()
            
        return saved_log

    async def update_previous_log_status(
        self,
        query: str,
        chat_model_id: int,
        new_status: str = "REQUERY",
        time_window_hours: int = 1
    ) -> bool:
        """
        이전 채팅 로그의 상태를 업데이트합니다 (Requery 시 사용).
        
        Args:
            query: 사용자 질문
            chat_model_id: 챗봇 모델 ID
            new_status: 변경할 상태 (기본값: REQUERY)
            time_window_hours: 검색할 시간 범위 (기본값: 1시간)
            
        Returns:
            업데이트 성공 여부
        """
        from datetime import timedelta
        
        # 검색 시간 제한 설정 (최근 N시간)
        time_limit = datetime.now() - timedelta(hours=time_window_hours)
        
        # 이전 로그 검색 (모델 ID 무시하고 쿼리만 일치하면 됨)
        # 사용자가 모델을 변경해서 재질문할 수도 있으므로 모델 ID 체크는 제외
        previous_log = await self.repository.find_latest_by_query(
            query=query,
            chat_model_id=None,  # 모델 ID 무시
            time_limit=time_limit
        )
        
        if previous_log:
            # 상태 업데이트
            await self.repository.update_status(previous_log.id, new_status)
            return True
            
        return False
