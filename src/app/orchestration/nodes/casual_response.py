"""일상 대화 응답 노드"""
import logging
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from .utils import _get_service_from_config, get_llm_mini, invoke_llm_with_resilience

logger = logging.getLogger(__name__)


async def casual_response_node(state: Union[ChatState, dict], config: dict) -> dict:
    """5-1단계: 일상 대화 응답 (가벼운 모델 사용)
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    user_message = chat_state.user_message
    chat_history = chat_state.chat_history
    user_id = chat_state.user_id
    
    # 이전 대화 맥락 포함
    messages = []
    if chat_history:
        for i, msg in enumerate(chat_history[-4:]):  # 최근 2턴
            # dict 형식인 경우
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    logger.warning(f"Casual response: 알 수 없는 role={role}, 메시지 스킵")
            # LangChain 메시지 객체인 경우 (BaseMessage)
            elif isinstance(msg, BaseMessage):
                if isinstance(msg, HumanMessage):
                    messages.append(msg)  # 그대로 사용
                elif isinstance(msg, AIMessage):
                    messages.append(msg)  # 그대로 사용
                else:
                    # 다른 BaseMessage 타입은 content 추출하여 HumanMessage로 변환
                    content = getattr(msg, "content", str(msg))
                    messages.append(HumanMessage(content=content))
            else:
                logger.warning(f"Casual response: 히스토리[{i}]가 dict도 BaseMessage도 아님, 타입={type(msg)}, 스킵")
    
    # 현재 질문 추가
    messages.append(HumanMessage(content=user_message))
    
    # 페르소나 설정 적용
    settings = chat_state.chatbot_settings
    persona = settings.get("persona_description") if settings else None
    
    if persona:
        base_prompt = persona
    else:
        base_prompt = """당신은 친절한 챗봇입니다.
사용자와 자연스럽고 친근하게 대화하세요.
간단하고 명확하게 답변하세요.
이전 대화 내용을 참고하여 맥락을 이해하고 답변하세요."""

    system_prompt = f"""{base_prompt}

**응답 길이 제한 (매우 중요):**
- 답변은 반드시 500자 이내로 작성하세요.
- 카카오 워크 blockit 텍스트 제한으로 인해 500자를 초과하면 답변이 표시되지 않습니다.
- 핵심 내용만 간결하게 전달하세요."""
    
    try:
        llm = get_llm_mini(config)
        # Circuit Breaker를 통해 LLM 호출 (장애 복구)
        final_messages = [SystemMessage(content=system_prompt)] + messages
        
        # 비동기 호출로 변경
        from .utils import ainvoke_llm_with_resilience
        response = await ainvoke_llm_with_resilience(llm, final_messages, config)
        
        # 실제 사용된 모델 ID 확인 (Haiku 또는 Sonnet)
        provider = _get_service_from_config(config, "provider")
        actual_model = provider.casual_model
        # 모델 ID에서 모델 이름 추출 (예: "anthropic.claude-haiku-4-5-20251001-v1:0" -> "claude-haiku-4-5")
        if "haiku" in actual_model.lower():
            model_name = "claude-haiku-4-5"
        elif "sonnet" in actual_model.lower():
            model_name = "claude-sonnet-4-5"
        else:
            model_name = "claude-haiku-4-5"  # 기본값
        
        logger.info(f"Casual response generated: user={user_id}, model={model_name}")
        
        # 히스토리 업데이트: 현재 대화(질문+답변)를 히스토리에 추가
        current_history = list(chat_state.chat_history) if chat_state.chat_history else []
        
        # 현재 질문과 답변을 히스토리에 추가 (dict 형식으로 통일)
        final_answer = response.content.strip()
        current_history.append({"role": "user", "content": user_message})
        current_history.append({"role": "assistant", "content": final_answer})
        
        # 최근 3턴(6개 메시지)만 유지 (메모리 최적화)
        if len(current_history) > 6:
            current_history = current_history[-6:]
        
        updated_state = chat_state.update(
            final_answer=final_answer,
            chat_history=current_history,  # 업데이트된 히스토리 반환
            model_used=model_name,
            route=chat_state.route + " -> casual_response",
            token_usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
        )
        return to_state_dict(updated_state)
        
    except Exception as e:
        logger.error(f"Casual response error: {e}")
        updated_state = chat_state.update(
            final_message="죄송합니다. 일시적인 오류가 발생했습니다.",
            route=chat_state.route + " -> casual_response_error"
        )
        return to_state_dict(updated_state)

