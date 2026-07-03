"""세션 관리 포트 인터페이스"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class SessionPort(ABC):
    """세션 관리 포트 인터페이스
    
    세션 및 대화 히스토리 관리를 추상화하는 포트입니다.
    도메인 레이어는 이 인터페이스를 통해 세션에 접근합니다.
    """
    
    @abstractmethod
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """세션 히스토리 조회 (동기)
        
        Args:
            session_id: 세션 ID
        
        Returns:
            대화 히스토리 리스트 (최근 3턴)
        """
        pass
    
    @abstractmethod
    async def get_history_async(self, session_id: str) -> List[Dict[str, Any]]:
        """세션 히스토리 조회 (비동기)
        
        Args:
            session_id: 세션 ID
        
        Returns:
            대화 히스토리 리스트 (최근 3턴)
        """
        pass
    
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """메시지 추가
        
        Args:
            session_id: 세션 ID
            role: 메시지 역할 ('user' | 'assistant' | 'system')
            content: 메시지 내용
        """
        pass
    
    @abstractmethod
    async def add_message_async(self, session_id: str, role: str, content: str) -> None:
        """메시지 추가 (비동기)
        
        Args:
            session_id: 세션 ID
            role: 메시지 역할 ('user' | 'assistant' | 'system')
            content: 메시지 내용
        """
        pass
    
    @abstractmethod
    def save_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """세션 히스토리 저장
        
        Args:
            session_id: 세션 ID
            history: 대화 히스토리 리스트
        """
        pass
    
    @abstractmethod
    async def save_history_async(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """세션 히스토리 저장 (비동기)
        
        Args:
            session_id: 세션 ID
            history: 대화 히스토리 리스트
        """
        pass

