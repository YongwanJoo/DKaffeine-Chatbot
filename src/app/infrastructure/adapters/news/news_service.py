"""뉴스 서비스 구현"""
import logging
from typing import List
from datetime import date, datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.domain.ports.news_port import NewsPort, NewsArticle
from app.infrastructure.persistence.models.article_model import Article
from app.infrastructure.persistence.session import db_session

logger = logging.getLogger(__name__)


class NewsService(NewsPort):
    """뉴스 서비스 구현체
    
    데이터베이스에서 오늘 날짜의 뉴스 기사를 조회합니다.
    
    Example:
        ```python
        news_service = NewsService()
        articles = news_service.get_today_news()
        ```
    """
    
    def get_today_news(self) -> List[NewsArticle]:
        """오늘 날짜의 뉴스 기사 조회
        
        Returns:
            오늘 날짜(published_at)에 해당하는 뉴스 기사 리스트
            오늘 뉴스가 없으면 최근 수집된 뉴스(최근 수집된 것) 반환
        """
        try:
            with db_session() as db:
                # 오늘 날짜 (시간 부분 제거)
                today = date.today()
                today_start = datetime.combine(today, datetime.min.time())
                today_end = datetime.combine(today, datetime.max.time())
                
                # 오늘 날짜의 기사 조회 (published_at 기준, 내림차순 정렬)
                articles = db.query(Article).filter(
                    and_(
                        Article.published_at >= today_start,
                        Article.published_at <= today_end,
                        Article.deleted_at.is_(None)  # 삭제되지 않은 기사만
                    )
                ).order_by(Article.published_at.desc()).all()
                
                # 오늘 뉴스가 없으면 최근 수집된 뉴스 조회 (created_at 기준, 최근 3개)
                if not articles:
                    logger.info("오늘 날짜 뉴스 없음, 최근 수집된 뉴스 조회")
                    articles = db.query(Article).filter(
                        Article.deleted_at.is_(None)  # 삭제되지 않은 기사만
                    ).order_by(Article.created_at.desc()).limit(1).all()  # 최근 수집된 3개
                
                # NewsArticle DTO로 변환
                result = []
                for article in articles:
                    # published_at을 date로 변환
                    published_date = None
                    if article.published_at:
                        published_date = article.published_at.date()
                    
                    news_article = NewsArticle(
                        id=article.id,
                        title=article.title,
                        description=article.description or "",
                        url=article.url,
                        image_url=article.image_url,
                        source=article.source,
                        published_at=published_date
                    )
                    result.append(news_article)
                
                logger.info(f"오늘 뉴스 조회 완료: {len(result)}건")
                return result
                
        except Exception as e:
            logger.error(f"뉴스 조회 중 오류 발생: {e}", exc_info=True)
            # 에러 발생 시 빈 리스트 반환 (서비스 안정성)
            return []

