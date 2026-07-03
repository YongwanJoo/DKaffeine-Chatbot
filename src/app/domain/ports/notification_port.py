"""알림 포트 정의"""
from abc import ABC, abstractmethod


class NotificationPort(ABC):
    """알림 발송을 위한 포트 인터페이스"""
    
    @abstractmethod
    async def send_faq_pending_notification(self, faq_id: int) -> bool:
        """FAQ 승인 대기 알림 발송
        
        Args:
            faq_id: 생성된 FAQ의 ID
            
        Returns:
            성공 여부
        """
        pass
