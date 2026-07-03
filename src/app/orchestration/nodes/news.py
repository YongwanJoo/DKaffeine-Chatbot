"""뉴스 요약 노드"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from langchain_core.messages import SystemMessage, HumanMessage
from .utils import _get_service_from_config, get_llm_mini, invoke_llm_with_resilience, ainvoke_llm_with_resilience
from app.domain.ports import NewsPort

logger = logging.getLogger(__name__)


async def news_summary_node(state: Union[ChatState, dict], config: dict) -> dict:
    """뉴스 요약 노드: 오늘의 뉴스를 조회하고 LLM으로 요약 생성
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    user_message = chat_state.user_message
    user_id = chat_state.user_id
    
    # config에서 서비스 가져오기
    news_service: NewsPort = _get_service_from_config(config, "news_service")
    provider = _get_service_from_config(config, "provider")
    
    try:
        # 오늘의 뉴스 조회 (비동기 처리)
        import asyncio
        articles = await asyncio.to_thread(news_service.get_today_news)
        
        if not articles:
            # 뉴스가 없을 때
            final_answer = "오늘 수집된 뉴스가 없습니다. 내일 다시 확인해주세요."
            logger.info(f"뉴스 요약: 뉴스 없음, user={user_id}")
            
            updated_state = chat_state.update(
                final_answer=final_answer,
                final_sources=["News"],
                model_used="claude-haiku-4-5",
                route=chat_state.route + " -> news_summary (no_news)",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            return to_state_dict(updated_state)
        
        # 뉴스 기사들을 텍스트로 구성
        news_texts = []
        for i, article in enumerate(articles, 2):
            news_text = f"[{i}] {article.title}\n"
            if article.description:
                news_text += f"   {article.description}\n"
            if article.url:
                news_text += f"   URL: {article.url}\n"
            if article.source:
                news_text += f"   출처: {article.source}\n"
            news_texts.append(news_text)
        
        news_content = "\n".join(news_texts)
        
        # LLM으로 뉴스 요약 생성
        system_prompt = """당신은 IT/개발 뉴스 요약 전문가입니다.
사용자에게 오늘의 주요 IT/개발 뉴스를 간결하고 이해하기 쉽게 요약해주세요.

**필수 시작 문구:**
- 답변의 첫 줄은 반드시 "오늘의 주요 IT 뉴스를 요약해드립니다."로 시작하세요.

요약 형식:
- 최대 2개의 주요 뉴스만 선택하여 요약하세요.
- **각 뉴스 항목의 구성:**
  1. 핵심 내용 요약 (2~3문장, 기술적 내용 쉽게 설명)
  2. 바로 다음 줄에 "출처: [URL]" 표기
  (예시:
   ...요약 내용입니다.
   출처: https://example.com)
- 전체적으로 흐름이 자연스럽게 연결되도록 작성
- "참고 문서"나 "연관 검색어" 같은 섹션 헤더를 절대 추가하지 마세요.
- 마크다운 형식(#, ##, ### 등)을 사용하지 마세요. 일반 텍스트로만 작성하세요
- 번호는 "1.", "2." 형식으로만 사용하세요

**응답 길이 제한 (매우 중요):**
- 답변은 반드시 500자 이내로 작성하세요.
- 카카오 워크 blockit 텍스트 제한으로 인해 500자를 초과하면 답변이 표시되지 않습니다.
- 2개의 뉴스를 합쳐서 총 500자 이내로 작성하세요.
- 각 뉴스는 핵심 내용만 간결하게 요약하세요.

한국어로 답변하세요."""
        
        user_prompt = f"""오늘의 IT/개발 뉴스 {len(articles)}건이 있습니다:

{news_content}

위 뉴스 중에서 가장 중요한 2개의 뉴스만 선택하여 요약해주세요.
2개의 뉴스를 합쳐서 총 500자 이내로 작성하세요."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 뉴스 요약은 긴 텍스트가 필요하므로 max_tokens를 충분히 설정
        # 경량 모델 사용 (casual 모델) + max_tokens 8192로 설정
        llm = provider.get_chat_llm(role="casual", max_tokens=8192)
        
        # 비동기 호출로 변경
        response = await ainvoke_llm_with_resilience(llm, messages, config)
        
        # 실제 사용된 모델 ID 확인
        actual_model = provider.casual_model
        if "haiku" in actual_model.lower():
            model_name = "claude-haiku-4-5"
        elif "sonnet" in actual_model.lower():
            model_name = "claude-sonnet-4-5"
        else:
            model_name = "claude-haiku-4-5"
        
        final_answer = response.content.strip()
        
        logger.info(f"뉴스 요약 생성 완료: user={user_id}, articles={len(articles)}, model={model_name}")
        
        # 히스토리 업데이트
        current_history = list(chat_state.chat_history) if chat_state.chat_history else []
        current_history.append({"role": "user", "content": user_message})
        current_history.append({"role": "assistant", "content": final_answer})
        
        # 최근 3턴(6개 메시지)만 유지
        if len(current_history) > 6:
            current_history = current_history[-6:]
        
        updated_state = chat_state.update(
            final_answer=final_answer,
            final_sources=[],  # 빈 리스트로 설정하여 state_service에서 참고 문서 섹션 추가 방지
            chat_history=current_history,
            model_used=model_name,
            route=chat_state.route + " -> news_summary",
            token_usage={"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800}
        )
        return to_state_dict(updated_state)
        
    except Exception as e:
        logger.error(f"뉴스 요약 중 오류 발생: {e}", exc_info=True)
        updated_state = chat_state.update(
            final_message="죄송합니다. 뉴스 요약 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            route=chat_state.route + " -> news_summary_error"
        )
        return to_state_dict(updated_state)

