from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class NewsArticleResponse(BaseModel):
    """뉴스 기사 응답 모델"""
    id: int
    title: str
    description: str
    url: str
    image_url: Optional[str] = None
    source: str
    published_at: Optional[date] = None

class NewsSummaryResponse(BaseModel):
    """뉴스 요약 응답 모델 (상세 화면용)"""
    short_summary: str  # Block Kit용 짧은 요약 (500자 이내)
    summary: str        # 상세 브리핑 (Markdown)
    articles: List[NewsArticleResponse]
    total_count: int
