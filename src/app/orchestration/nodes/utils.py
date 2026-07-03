"""공통 유틸리티 함수"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict

logger = logging.getLogger(__name__)


def _get_service_from_config(config: dict, service_name: str):
    """
    config에서 서비스를 가져옴 (포트 인터페이스 타입 반환)
    
    Args:
        config: LangGraph config 딕셔너리 (필수)
        service_name: 서비스 이름 (예: "guardrail_service")
    
    Returns:
        서비스 인스턴스 (포트 인터페이스 타입)
    
    Raises:
        ValueError: config가 없거나 서비스가 없을 때
    """
    if not config or not isinstance(config, dict):
        logger.error(f"config가 dict가 아닙니다. type: {type(config)}, value: {config}")
        raise ValueError(
            f"config가 제공되지 않았습니다. {service_name}을 사용하려면 "
            "messages.py에서 app.state.services를 config로 전달해야 합니다."
        )
    
    
    service = config.get(service_name)
    if service is None:
        logger.error(f"config에 {service_name}이 없습니다. config 키: {list(config.keys()) if isinstance(config, dict) else 'not dict'}")
        raise ValueError(
            f"config에 {service_name}이 없습니다. "
            f"현재 config 키: {list(config.keys()) if isinstance(config, dict) else 'not dict'}. "
            "app_factory.py의 create_services()에서 모든 서비스를 생성하고 "
            "messages.py에서 config로 전달해야 합니다."
        )
    
    return service


def get_llm_mini(config: dict):
    """일상 대화용 경량 LLM (지연 초기화)"""
    provider_instance = _get_service_from_config(config, "provider")
    if not hasattr(get_llm_mini, '_instance'):
        get_llm_mini._instance = provider_instance.get_chat_llm(role="casual")
    return get_llm_mini._instance


def invoke_llm_with_resilience(llm, messages, config: dict):
    """Circuit Breaker를 통해 LLM 호출 (장애 복구)
    
    Args:
        llm: LangChain LLM 인스턴스
        messages: LLM에 전달할 메시지 리스트
        config: LangGraph config (provider 포함, 필수)
    
    Returns:
        LLM 응답 객체
    """
    provider_instance = _get_service_from_config(config, "provider")
    if hasattr(provider_instance, 'invoke_with_resilience'):
        return provider_instance.invoke_with_resilience(llm, messages)
    else:
        logger.warning("LLMProvider의 invoke_with_resilience를 사용할 수 없습니다. 직접 호출합니다.")
        return llm.invoke(messages)


async def ainvoke_llm_with_resilience(llm, messages, config: dict):
    """Circuit Breaker를 통해 LLM 비동기 호출 (장애 복구)
    
    Args:
        llm: LangChain LLM 인스턴스
        messages: LLM에 전달할 메시지 리스트
        config: LangGraph config (provider 포함, 필수)
    
    Returns:
        LLM 응답 객체
    """
    provider_instance = _get_service_from_config(config, "provider")
    if hasattr(provider_instance, 'ainvoke_with_resilience'):
        return await provider_instance.ainvoke_with_resilience(llm, messages)
    else:
        logger.warning("LLMProvider의 ainvoke_with_resilience를 사용할 수 없습니다. 직접 호출합니다.")
        if hasattr(llm, "ainvoke"):
            return await llm.ainvoke(messages)
        else:
            import asyncio
            return await asyncio.to_thread(llm.invoke, messages)


def _get_detailed_block_reason(category: str, default_reason: str, details: dict) -> str:
    """카테고리별 구체적인 차단 이유 생성"""
    # Blacklist 카테고리 매핑 (blacklist.json의 카테고리 → 표준 카테고리)
    blacklist_category_map = {
        "profanity": "profanity",
        "sexual_content": "sexual",
        "violence": "violence",
        "personal_info_requests": "personal_info",
        "spam": "spam",
    }
    
    # 표준 카테고리로 변환
    standard_category = blacklist_category_map.get(category, category)
    
    category_messages = {
        "profanity": "부적절한 언어가 감지되었습니다. 업무 관련 질문만 답변 가능합니다.",
        "sexual": "부적절한 내용이 감지되었습니다. 업무 관련 질문만 답변 가능합니다.",
        "violence": "폭력적이거나 위협적인 내용이 감지되었습니다. 업무 관련 질문만 답변 가능합니다.",
        "personal_info": "개인정보 요청이 감지되었습니다. 업무 관련 질문만 답변 가능합니다.",
        "spam": "스팸 또는 광고성 내용이 감지되었습니다. 업무 관련 질문만 답변 가능합니다.",
        "off_topic": "업무와 무관한 주제입니다. 업무 관련 질문만 답변 가능합니다.",
    }
    
    # 카테고리에 따른 메시지 반환
    if standard_category in category_messages:
        return category_messages[standard_category]
    
    # off_topic 판단 (details에 business_related가 false인 경우)
    if details and details.get("business_related") is False:
        return "업무와 무관한 주제입니다. 업무 관련 질문만 답변 가능합니다."
    
    # 기본 메시지
    return default_reason or "업무 관련 질문만 답변 가능합니다."

