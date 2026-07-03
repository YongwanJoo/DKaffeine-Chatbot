"""세션 관리 어댑터"""
from typing import List, Dict, Any, Optional

from app.domain.ports.session_port import SessionPort
from app.domain.services.session_service import SessionService


class SessionAdapter(SessionPort):
    """세션 관리 어댑터 구현
    
    SessionPort 인터페이스를 구현하여 세션 관리를 제공합니다.
    """
    
    def __init__(self, session_service: Optional[SessionService] = None):
        """세션 관리 어댑터 초기화
        
        Args:
            session_service: SessionService 인스턴스 (None이면 클래스 메서드 사용)
        """
        self._service = session_service
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """세션 히스토리 조회 (동기)"""
        if self._service:
            return self._service.get_history(session_id)
        return SessionService.get_history(session_id)
    
    async def get_history_async(self, session_id: str) -> List[Dict[str, Any]]:
        """세션 히스토리 조회 (비동기)"""
        if self._service:
            return await self._service.get_history_async(session_id)
        return await SessionService.get_history_async(session_id)
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """메시지 추가 (동기)"""
        if self._service:
            self._service.add_message(session_id, role, content)
        else:
            SessionService.add_message(session_id, role, content)
    
    async def add_message_async(self, session_id: str, role: str, content: str) -> None:
        """메시지 추가 (비동기)"""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.add_message, session_id, role, content)
    
    def save_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """세션 히스토리 저장 (동기)"""
        if self._service:
            self._service.save_history(session_id, history)
        else:
            SessionService.save_history(session_id, history)
    
    async def save_history_async(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """세션 히스토리 저장 (비동기)"""
        if self._service:
            await self._service.save_history_async(session_id, history)
        else:
            await SessionService.save_history_async(session_id, history)

