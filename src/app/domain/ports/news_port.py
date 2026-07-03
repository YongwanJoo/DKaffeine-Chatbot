"""뉴스 서비스 포트 (인터페이스)"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import date


class NewsArticle:
    """뉴스 기사 DTO"""
    def __init__(
        self,
        id: Optional[int],
        title: str,
        description: str,
        url: str,
        image_url: Optional[str],
        source: Optional[str],
        published_at: Optional[date]
    ):
        self.id = id
        self.title = title
        self.description = description
        self.url = url
        self.image_url = image_url
        self.source = source
        self.published_at = published_at


class NewsPort(ABC):
    """뉴스 서비스 포트 (인터페이스)
    
    뉴스 데이터 조회 서비스의 추상 인터페이스입니다.
    모든 뉴스 서비스 구현체는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def get_today_news(self) -> List[NewsArticle]:
        """오늘 날짜의 뉴스 기사 조회
        
        Returns:
            오늘 날짜(published_at)에 해당하는 뉴스 기사 리스트
            뉴스가 없으면 빈 리스트 반환
        """
        pass

