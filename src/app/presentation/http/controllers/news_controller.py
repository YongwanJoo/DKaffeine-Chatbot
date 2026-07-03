from fastapi import APIRouter, Depends, HTTPException
import logging

from app.presentation.http.models.news_models import NewsSummaryResponse
from app.domain.services.news_usecase import NewsUsecase
from app.infrastructure.adapters.news.news_service import NewsService
from app.infrastructure.adapters.llm import LLMProvider
from app.infrastructure.adapters.cache.cache_service import CacheService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/news", tags=["news"])

def get_news_usecase() -> NewsUsecase:
    """NewsUsecase 의존성 주입"""
    news_service = NewsService()
    # LLMProvider는 내부적으로 Redis 등을 사용할 수 있으므로 필요시 주입
    # 여기서는 기본 초기화 사용
    llm_provider = LLMProvider()
    # CacheService 주입 (Redis 사용 여부는 내부 설정에 따름)
    cache_service = CacheService(use_redis=True)
    return NewsUsecase(news_service, llm_provider, cache_service)

@router.post("/today/process", response_model=NewsSummaryResponse)
async def process_today_news(
    usecase: NewsUsecase = Depends(get_news_usecase)
):
    """오늘의 뉴스 수집 및 처리 (AI 서버 트리거용)
    
    뉴스 수집 서버가 수집 완료 후 이 API를 호출하여
    AI 요약 생성 및 Redis 캐싱을 수행합니다.
    """
    try:
        logger.info("뉴스 처리 요청 수신")
        return await usecase.process_today_news()
    except Exception as e:
        logger.error(f"뉴스 처리 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="뉴스 처리 중 오류가 발생했습니다.")

@router.get("/today/summary", response_model=NewsSummaryResponse)
async def get_today_news_summary(
    usecase: NewsUsecase = Depends(get_news_usecase)
):
    """오늘의 뉴스 상세 요약 조회 (인앱 브라우저용)
    
    오늘 수집된 모든 IT 뉴스를 종합하여 상세 브리핑을 생성하고,
    전체 기사 목록과 함께 반환합니다.
    Redis 캐시가 있으면 캐시된 데이터를 반환합니다.
    """
    try:
        return await usecase.get_today_news_summary()
    except Exception as e:
        logger.error(f"뉴스 상세 요약 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="뉴스 정보를 가져오는 중 오류가 발생했습니다.")
