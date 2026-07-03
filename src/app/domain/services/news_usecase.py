import logging
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.domain.ports.news_port import NewsPort
from app.infrastructure.adapters.llm import LLMProvider
from app.presentation.http.models.news_models import NewsSummaryResponse, NewsArticleResponse

logger = logging.getLogger(__name__)

import json
from datetime import datetime

class NewsUsecase:
    """뉴스 관련 비즈니스 로직"""
    
    def __init__(self, news_service: NewsPort, llm_provider: LLMProvider, cache_service: Any = None):
        self.news_service = news_service
        self.llm_provider = llm_provider
        # 순환 참조 방지를 위해 Any로 타입 힌팅하거나 TYPE_CHECKING 사용
        # 여기서는 간단히 Any로 처리하고 런타임에 주입
        self.cache_service = cache_service
    
    async def process_today_news(self) -> NewsSummaryResponse:
        """오늘의 뉴스 수집 및 처리 (AI 요약 + Redis 저장)"""
        # 1. 뉴스 조회
        articles = self.news_service.get_today_news()
        
        if not articles:
            return NewsSummaryResponse(
                summary="오늘 수집된 뉴스가 없습니다.",
                articles=[],
                total_count=0
            )
        
        # 2. LLM을 사용한 종합 요약 생성
        # 뉴스 텍스트 구성
        news_texts = []
        for i, article in enumerate(articles, 1):
            news_text = f"[{i}] {article.title}\n"
            if article.description:
                news_text += f"   {article.description}\n"
            if article.source:
                news_text += f"   출처: {article.source}\n"
            news_texts.append(news_text)
        
        news_content = "\n".join(news_texts)
        
        system_prompt = """당신은 IT/개발 뉴스 전문 에디터입니다.
오늘의 주요 IT 뉴스를 종합적으로 분석하여 두 가지 버전의 브리핑을 작성해주세요.

작성 가이드:
1. **short_summary (500자 이내)**:
   - 바쁜 개발자를 위한 핵심 요약입니다.
   - 주요 이슈 3가지를 글머리 기호로 간결하게 정리하세요.
   - 전체적인 트렌드를 한 문장으로 요약하세요.
   - 이모지를 적절히 사용하여 가독성을 높이세요.

2. **full_summary (상세 브리핑)**:
   - 심층적인 분석이 담긴 상세 리포트입니다.
   - **전체 요약**: 오늘 뉴스의 주요 흐름과 트렌드를 3~4문장으로 서술하세요.
   - **주요 이슈**: 가장 중요한 3~4가지 이슈를 선정하여 각각 소제목과 함께 상세히 설명하세요.
   - **업계 동향**: 기술, 기업, 시장 관점에서의 변화를 분석하세요.
   - **마무리**: 오늘의 뉴스가 주는 시사점을 한 문장으로 정리하세요.
   - 마크다운 형식을 사용하여 가독성 있게 작성하세요.

출력 형식:
반드시 아래 JSON 형식으로만 응답하세요. 다른 말은 포함하지 마세요.
{
    "short_summary": "여기에 짧은 요약 작성...",
    "full_summary": "여기에 상세 브리핑 작성 (마크다운 포함)..."
}
"""
        
        user_prompt = f"""오늘의 IT/개발 뉴스 {len(articles)}건입니다:

{news_content}

위 내용을 바탕으로 JSON 형식의 브리핑을 작성해주세요."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # LLM 호출
        short_summary = "요약 생성 실패"
        full_summary = "뉴스 요약 생성 중 오류가 발생했습니다. 아래 기사 목록을 참고해주세요."
        
        try:
            llm = self.llm_provider.get_chat_llm(role="business", max_tokens=4096)
            response = await llm.ainvoke(messages)
            content = response.content.strip()
            
            # JSON 파싱 시도
            try:
                # 코드 블록 제거 (혹시 LLM이 ```json ... ``` 으로 감쌀 경우)
                if "```" in content:
                    content = content.replace("```json", "").replace("```", "").strip()
                
                data = json.loads(content)
                short_summary = data.get("short_summary", short_summary)
                full_summary = data.get("full_summary", full_summary)
            except json.JSONDecodeError:
                logger.warning(f"LLM 응답 JSON 파싱 실패. 원본 텍스트 사용: {content[:100]}...")
                # 파싱 실패 시 전체 내용을 full_summary로 간주하고 short_summary는 앞부분만 사용
                full_summary = content
                short_summary = content[:500] + "..."
                
        except Exception as e:
            logger.error(f"뉴스 요약 생성 실패: {e}")
        
        # 3. 응답 모델 생성
        article_responses = [
            NewsArticleResponse(
                id=article.id,
                title=article.title,
                description=article.description,
                url=article.url,
                image_url=article.image_url,
                source=article.source,
                published_at=article.published_at
            )
            for article in articles
        ]
        
        response_model = NewsSummaryResponse(
            short_summary=short_summary,
            summary=full_summary,
            articles=article_responses,
            total_count=len(articles)
        )
        
        # 4. Redis 저장 (TTL 24시간)
        if self.cache_service:
            try:
                # Pydantic 모델을 dict로 변환 후 JSON 직렬화
                # datetime 객체 처리를 위해 json_encoders 사용 또는 default=str
                cache_value = response_model.json()
                self.cache_service.set_raw("news:today:summary", cache_value, ttl=86400)
                logger.info("뉴스 요약 Redis 저장 완료 (TTL 24h)")
            except Exception as e:
                logger.error(f"뉴스 요약 Redis 저장 실패: {e}")
        
        return response_model

    async def get_today_news_summary(self) -> NewsSummaryResponse:
        """오늘의 뉴스 상세 요약 조회"""
        # 1. Redis 조회
        if self.cache_service:
            cached = self.cache_service.get_raw("news:today:summary")
            if cached:
                try:
                    logger.info("뉴스 요약 Cache Hit")
                    return NewsSummaryResponse.parse_raw(cached)
                except Exception as e:
                    logger.error(f"캐시 데이터 파싱 실패: {e}")
        
        # 2. 없으면 처리 로직 호출 (Fallback)
        logger.info("뉴스 요약 Cache Miss - 새로 생성")
        return await self.process_today_news()
