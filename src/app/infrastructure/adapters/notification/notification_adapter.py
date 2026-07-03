"""알림 어댑터 구현"""
import logging
import httpx
from typing import Optional

from app.domain.ports.notification_port import NotificationPort
from app.domain.ports.config_port import ConfigPort

logger = logging.getLogger(__name__)


class NotificationAdapter(NotificationPort):
    """HTTP를 통한 알림 발송 어댑터"""
    
    def __init__(self, config: ConfigPort):
        """
        Args:
            config: 설정 포트
        """
        self.config = config
        # 백엔드 URL 설정 (app 섹션에서 조회, 전체 API 경로 포함)
        # 실제 백엔드 API: POST /api/notifications/faq-pending/{faqId}
        # 예: https://backend.dkaffeine.com/api/notifications/faq-pending
        self.backend_url = self.config.get(
            "backend_url", 
            "https://backend.dkaffeine.com/api/notifications/faq-pending", 
            section="app"
        ).rstrip("/")
        logger.info(f"NotificationAdapter 초기화: backend_url={self.backend_url}")
        
    async def send_faq_pending_notification(self, faq_id: int) -> bool:
        """FAQ 승인 대기 알림 발송
        
        POST {backend_url}/{faq_id}
        backend_url에는 전체 API 경로가 포함되어 있음
        """
        # backend_url에 이미 전체 경로가 포함되어 있으므로 FAQ ID만 추가
        url = f"{self.backend_url}/{faq_id}"
        
        try:
            logger.info(f"FAQ 알림 발송 시도: faq_id={faq_id}, url={url}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, timeout=10.0)
                
                if response.status_code == 200:
                    logger.info(f"FAQ 알림 발송 성공: faq_id={faq_id}, url={url}")
                    return True
                else:
                    logger.warning(
                        f"FAQ 알림 발송 실패: faq_id={faq_id}, "
                        f"status={response.status_code}, url={url}, body={response.text[:200]}"
                    )
                    return False
                    
        except httpx.ConnectError as e:
            logger.error(
                f"FAQ 알림 발송 연결 실패: faq_id={faq_id}, url={url}, "
                f"error={type(e).__name__}: {str(e)}"
            )
            return False
        except httpx.TimeoutException as e:
            logger.error(
                f"FAQ 알림 발송 타임아웃: faq_id={faq_id}, url={url}, "
                f"error={type(e).__name__}: {str(e)}"
            )
            return False
        except Exception as e:
            logger.error(
                f"FAQ 알림 발송 중 오류 발생: faq_id={faq_id}, url={url}, "
                f"error={type(e).__name__}: {str(e)}", exc_info=True
            )
            return False
