"""통합 분석 노드: Guardrail, Intent Analysis, Business Classification"""
import logging
import json
import re
from typing import Union
from ..state import ChatState, ensure_chat_state, to_state_dict
from app.domain.ports import GuardrailPort
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from .utils import _get_service_from_config, invoke_llm_with_resilience

logger = logging.getLogger(__name__)


async def unified_analysis_node(state: Union[ChatState, dict], config: dict) -> dict:
    """통합 분석 노드: Guardrail, Intent Analysis, Business Classification을 단일 LLM 호출로 수행
    
    성능 최적화: 3번의 순차 LLM 호출을 1번으로 통합하여 응답 시간을 50% 이상 단축
    
    LangGraph 호환성을 위해 dict를 반환합니다.
    """
    # dict를 ChatState로 변환 (타입 검증)
    chat_state = ensure_chat_state(state)
    
    # 이미 blacklist에서 차단되었으면 스킵
    if chat_state.blacklist_blocked:
        updated_state = chat_state.update(
            guardrail_passed=False,
            route=chat_state.route + " -> unified_analysis_skipped"
        )
        return to_state_dict(updated_state)
    
    user_message = chat_state.user_message
    user_id = chat_state.user_id
    chat_history = chat_state.chat_history
    chatbot_settings = chat_state.chatbot_settings
    
    # LLM Provider 가져오기
    provider = _get_service_from_config(config, "provider")
    guardrail_service: GuardrailPort = _get_service_from_config(config, "guardrail_service")
    
    # 대화 맥락 정보 구성
    context_info = ""
    is_casual_context = False
    
    if chat_history and len(chat_history) > 0:
        last_message = chat_history[-1]
        last_role = ""
        last_content = ""
        
        if isinstance(last_message, dict):
            last_role = last_message.get("role", "")
            last_content = last_message.get("content", "")
        else:
            if isinstance(last_message, BaseMessage):
                last_role = "assistant" if isinstance(last_message, AIMessage) else "user"
                last_content = getattr(last_message, "content", "")
        
        if last_role == "assistant" and last_content:
            # 타입 안전성: chatbot_settings가 dict가 아니면 빈 dict 사용
            if isinstance(chatbot_settings, dict):
                keywords_config = chatbot_settings.get("guardrail_keywords", {})
            else:
                keywords_config = {}
            # 타입 안전성: keywords_config가 dict가 아니면 빈 dict 사용
            if not isinstance(keywords_config, dict):
                keywords_config = {}
            casual_keywords = keywords_config.get("casual_keywords", [
                "맛집", "음식", "식당", "메뉴", "추천", "건강", "운동", "안녕", "감사"
            ])
            is_casual_context = any(keyword in last_content[:100] for keyword in casual_keywords)
            
            if is_casual_context:
                context_info = f"\n\n**대화 맥락:**\n봇의 이전 메시지 (일상 대화): {last_content[:200]}"
            else:
                context_info = f"\n\n**대화 맥락:**\n봇의 이전 답변: {last_content[:200]}"
    
    # 하이브리드 접근: 키워드 기반 빠른 분류 (LLM 호출 전)
    # 1단계: 키워드 매칭으로 80-85% 케이스 처리 (LLM 호출 스킵)
    # 타입 안전성: chatbot_settings가 dict가 아니면 빈 dict 사용
    if isinstance(chatbot_settings, dict):
        keywords_config = chatbot_settings.get("guardrail_keywords", {})
    else:
        keywords_config = {}
    # 타입 안전성: keywords_config가 dict가 아니면 빈 dict 사용
    if not isinstance(keywords_config, dict):
        keywords_config = {}
    
    # 업무 키워드 확장
    business_keywords = keywords_config.get("business_keywords", [
        "휴가", "연차", "신청", "취소", "방법", "절차", "출장", "복지", "복장", 
        "시스템", "도구", "규정", "정책", "업무", "회사", "사내", "포털",
        "근태", "출근", "퇴근", "병가", "반차", "경조사", "승인", "부서장",
        "출산", "육아", "보육", "수당", "급여", "인사", "채용", "평가",
        "센터", "부서", "팀", "조직", "커넥트", "설명", "뭐야", "무엇",
        "구성", "구조", "인력", "개발자", "디케이테크인", "어떻게 되나요", "어떤가요",
        "문서", "양식", "서류", "제출", "처리", "신고", "보고", "결재", "결제",
        "회의", "미팅", "일정", "스케줄", "예약",
        "교육", "연수", "세미나", "워크샵", "프로젝트", "과제", "업무지시",
        "보안", "권한", "접근", "로그인", "비밀번호", "계정", "인증"
    ])
    
    # 일상 대화 키워드 (날씨 제외 - 실시간 정보 제공 불가)
    casual_keywords = keywords_config.get("casual_keywords", [
        "맛집", "식당", "음식", "먹", "메뉴", "추천", "뭐 먹", "뭐 먹을까",
        "점심", "저녁", "식사", "밥", "한식", "중식", "양식", "일식",
        "건강", "운동", "다이어트", "건강한", "헬스", "요가", "필라테스",
        "안녕", "감사", "고마워", "반가워", "안녕하세요", "감사합니다", "고맙습니다",
        "수고", "수고하셨습니다", "수고하세요", "잘 지내", "잘 지냈어"
    ])
    
    # 단어 단위로 정확한 매칭 + 부분 문자열 매칭 (인사말 등 짧은 단어 처리)
    user_message_lower = user_message.lower()
    words = set(re.findall(r'\b\w+\b', user_message_lower))
    business_keywords_set = set(business_keywords)
    casual_keywords_set = set(casual_keywords)
    
    # 뉴스 키워드 체크 (최우선: "뉴뉴")
    news_keywords = ["뉴뉴"]
    has_news_keyword = any(kw in user_message_lower for kw in news_keywords)
    
    # 뉴스 키워드가 포함된 경우 강제로 news로 분류 (최우선, LLM 호출 전)
    if has_news_keyword:
        updated_state = chat_state.update(
            guardrail_passed=True,
            guardrail_confidence=0.95,
            intent_type="news",
            intent_confidence=0.95,
            intent_analysis_details={"reason": "뉴스 키워드 감지로 자동 분류"},
            intent_category=None,
            blocked=False,
            route=chat_state.route + " -> unified_analysis_passed (news_keyword_detected)",
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # LLM 호출 없음
        )
        return to_state_dict(updated_state)
    
    # 업무 키워드 체크 (단어 단위 + 부분 문자열 매칭)
    has_business_keyword_word = bool(words & business_keywords_set)
    has_business_keyword_substring = any(kw in user_message_lower for kw in business_keywords)
    has_business_keyword = has_business_keyword_word or has_business_keyword_substring
    
    # 일상 대화 키워드 체크 (단어 단위 + 부분 문자열 매칭)
    # "안녕" 같은 짧은 인사말도 매칭되도록 부분 문자열 매칭 추가
    has_casual_keyword_word = bool(words & casual_keywords_set)
    has_casual_keyword_substring = any(kw in user_message_lower for kw in casual_keywords)
    has_casual_keyword = (has_casual_keyword_word or has_casual_keyword_substring) and not has_business_keyword
    
    # 업무 키워드가 포함된 경우 강제로 business로 분류 (LLM 호출 전)
    if has_business_keyword:
        # 카테고리 자동 분류 (키워드 기반)
        category = None
        if any(kw in user_message_lower for kw in ["휴가", "연차", "반차", "병가", "경조사"]):
            category = "leave"
        elif any(kw in user_message_lower for kw in ["출산", "육아", "보육"]):
            category = "birth"
        elif any(kw in user_message_lower for kw in ["복지", "건강검진", "점심", "사내 시설"]):
            category = "welfare"
        elif any(kw in user_message_lower for kw in ["복장", "드레스코드", "출근 복장"]):
            category = "dress_code"
        elif any(kw in user_message_lower for kw in ["출장", "출장비"]):
            category = "travel"
        elif any(kw in user_message_lower for kw in ["시스템", "포털", "프로그램", "접속"]):
            category = "system"
        else:
            category = "general"
        
        updated_state = chat_state.update(
            guardrail_passed=True,
            guardrail_confidence=0.95,
            intent_type="business",
            intent_confidence=0.95,
            intent_analysis_details={"reason": "업무 키워드 감지로 자동 분류"},
            intent_category=category,
            blocked=False,
            route=chat_state.route + " -> unified_analysis_passed (keyword_detected)",
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # LLM 호출 없음
        )
        return to_state_dict(updated_state)
    
    # 일상 대화 키워드가 포함된 경우 강제로 casual로 분류 (LLM 호출 전)
    if has_casual_keyword:
        updated_state = chat_state.update(
            guardrail_passed=True,
            guardrail_confidence=0.95,
            intent_type="casual",
            intent_confidence=0.95,
            intent_analysis_details={"reason": "일상 대화 키워드 감지로 자동 분류"},
            intent_category=None,
            blocked=False,
            route=chat_state.route + " -> unified_analysis_passed (casual_keyword_detected)",
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # LLM 호출 없음
        )
        return to_state_dict(updated_state)
    
    # 2단계: 모호한 케이스만 Few-shot LLM 사용 (15-20%)
    # 통합 프롬프트 구성 (Few-shot 예제 포함)
    system_prompt = """당신은 B2B 업무용 챗봇의 통합 분석 시스템입니다. 다음 3가지 작업을 동시에 수행하세요:

1. **안전성 검증 (Guardrail)**: 사용자 질문이 적절한지 판단

2. **의도 분석 (Intent)**: 일상 대화인지 업무 질문인지 분류

3. **업무 카테고리 분류 (Category)**: 업무 질문인 경우 세부 카테고리 분류

**안전성 검증 규칙:**

- **항상 허용**: 맛집/음식 추천, 건강, 인사말, 모든 업무 관련 질문

- **차단**: 정치, 종교, 선정적/음란, 욕설/비속어, 혐오 표현

**의도 분석 규칙 (매우 중요):**

- **casual**: 인사말, 일반 대화, 감사 인사, 맛집/음식 추천, 건강 등 (업무 무관)

  * **주의**: "연차", "휴가", "신청", "방법", "절차", "시스템", "회사", "업무" 등이 포함된 질문은 절대 casual로 분류하지 마세요

- **business**: 회사 규정, 절차, 시스템 사용법, 업무 관련 질문, 제품/서비스 설명 요청

  * **필수**: "연차", "휴가", "신청", "방법", "절차", "시스템", "회사", "업무" 등이 포함되면 반드시 business로 분류

- **중요**: 

  * 봇이 먼저 일상 대화를 시작한 경우에만 사용자 응답을 casual로 분류

  * 업무 관련 키워드가 포함되면 문맥과 관계없이 반드시 business로 분류

  * 모호한 경우(예: "설명해줘"만 있는 경우)는 문맥을 고려하되, 제품/서비스명이 언급되면 business로 분류

**업무 카테고리 (business인 경우만):**

- **leave**: 휴가, 연차, 반차, 병가, 경조사 휴가

- **birth**: 출산 휴가, 육아 휴가, 배우자 출산 휴가

- **welfare**: 복리후생, 건강검진, 점심 식사, 사내 시설

- **dress_code**: 복장 규정, 출근 복장, 드레스코드

- **travel**: 출장, 출장비, 출장 절차

- **system**: 사내 시스템 사용법, 포털 접속, 프로그램 사용

- **general**: 기타 일반적인 업무 질문

**응답 형식 (반드시 JSON):**

{
  "is_safe": true | false,
  "intent": "casual" | "business" | "unsafe",
  "category": "leave" | "birth" | "welfare" | "dress_code" | "travel" | "system" | "general" | null,
  "confidence": 0.0-1.0,
  "reason": "판단 사유 (한국어)"
}

**참고:**

- intent가 "unsafe"이면 is_safe는 false

- intent가 "casual"이면 category는 null

- intent가 "business"이면 category는 위의 카테고리 중 하나"""
    
    prompt_content = f"다음 질문을 통합 분석하세요:{context_info}\n\n사용자 질문: {user_message}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content)
    ]
    
    try:
        # LLM 인스턴스 가져오기 (경량 모델 사용)
        llm = provider.get_chat_llm(role="casual")
        
        # Circuit Breaker를 통해 LLM 호출 (장애 복구)
        # 비동기 호출로 변경
        from .utils import ainvoke_llm_with_resilience
        response = await ainvoke_llm_with_resilience(llm, messages, config)
        content = response.content.strip()
        
        # JSON 추출
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content, strict=False)
        
        is_safe = result.get("is_safe", True)
        intent = result.get("intent", "business")
        category = result.get("category")
        confidence = result.get("confidence", 0.5)
        reason = result.get("reason", "")
        
        # 차단된 경우
        if not is_safe or intent == "unsafe":
            logger.warning(
                f"Unified analysis blocked: user={user_id}, "
                f"intent={intent}, confidence={confidence}"
            )
            
            # guardrail_config.json의 표준 메시지 사용 (형식 통일)
            guardrail_service: GuardrailPort = _get_service_from_config(config, "guardrail_service")
            # GuardrailService 구현체의 config 속성 접근 (타입 체크)
            if hasattr(guardrail_service, 'config'):
                messages_config = guardrail_service.config.get("response_messages", {})
            else:
                messages_config = {}
            
            # 카테고리 매핑 (LLM 응답 카테고리 → 설정 파일 키)
            # LLM이 반환한 category가 없으면 reason에서 추출 시도
            category_from_reason = None
            if reason:
                reason_lower = reason.lower()
                if "욕설" in reason or "비속어" in reason or "profanity" in reason_lower:
                    category_from_reason = "profanity"
                elif "선정" in reason or "음란" in reason or "sexual" in reason_lower:
                    category_from_reason = "sexual"
                elif "폭력" in reason or "violence" in reason_lower:
                    category_from_reason = "violence"
                elif "개인정보" in reason or "personal" in reason_lower:
                    category_from_reason = "personal_info"
                elif "스팸" in reason or "spam" in reason_lower:
                    category_from_reason = "spam"
            
            # 카테고리 매핑 (LLM 응답 카테고리 → 설정 파일 키)
            category_map = {
                "profanity": "blocked_profanity",
                "sexual": "blocked_sexual",
                "violence": "blocked_violence",
                "personal_info": "blocked_personal_info",
                "spam": "blocked_spam",
            }
            
            # category 우선순위: LLM 응답 category > reason에서 추출한 category > general
            # unified_analysis_node의 LLM 응답에는 category가 없으므로 reason에서 추출
            detected_category = category_from_reason or "general"
            config_key = category_map.get(detected_category, "blocked_general")
            detailed_reason = messages_config.get(config_key, messages_config.get("blocked_general", "부적절한 내용이 감지되었습니다. 업무 관련 질문만 답변 가능합니다."))
            
            updated_state = chat_state.update(
                guardrail_passed=False,
                guardrail_reason=detailed_reason,
                guardrail_category=detected_category,
                guardrail_confidence=confidence,
                intent_type=intent,
                blocked=True,
                block_reason=detailed_reason,
                final_message=detailed_reason,
                route=chat_state.route + " -> unified_analysis_failed",
                token_usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
            )
            return to_state_dict(updated_state)
        
        # 통과한 경우
        
        # 결과를 state에 반영
        intent_analysis_details = {"reason": reason}
        if intent == "business" and category:
            intent_analysis_details["business_category"] = category
            intent_analysis_details["category_reason"] = reason
        
        updated_state = chat_state.update(
            guardrail_passed=True,
            guardrail_confidence=confidence,
            intent_type=intent,
            intent_confidence=confidence,
            intent_analysis_details=intent_analysis_details,
            intent_category=category if intent == "business" else None,
            blocked=False,
            route=chat_state.route + " -> unified_analysis_passed",
            token_usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
        )
        return to_state_dict(updated_state)
        
    except Exception as e:
        logger.error(f"Unified analysis error: {e}", exc_info=True)
        
        # 에러 발생 시 키워드 기반으로 재시도
        user_message_lower = user_message.lower()
        
        # 키워드 기반 빠른 분류 (에러 발생 시 폴백)
        # 타입 안전성: chatbot_settings가 dict가 아니면 빈 dict 사용
        if isinstance(chatbot_settings, dict):
            keywords_config = chatbot_settings.get("guardrail_keywords", {})
        else:
            keywords_config = {}
        # 타입 안전성: keywords_config가 dict가 아니면 빈 dict 사용
        if not isinstance(keywords_config, dict):
            keywords_config = {}
        business_keywords = keywords_config.get("business_keywords", [
            "휴가", "연차", "신청", "취소", "방법", "절차", "출장", "복지", "복장", 
            "시스템", "도구", "규정", "정책", "업무", "회사", "사내", "포털"
        ])
        casual_keywords = keywords_config.get("casual_keywords", [
            "맛집", "식당", "음식", "먹", "메뉴", "추천", "점심", "저녁", "식사", "밥",
            "건강", "운동", "안녕", "감사", "고마워", "반가워", "안녕하세요", "감사합니다"
        ])
        
        # 부분 문자열 매칭으로 키워드 확인
        has_business_keyword = any(kw in user_message_lower for kw in business_keywords)
        has_casual_keyword = any(kw in user_message_lower for kw in casual_keywords) and not has_business_keyword
        
        # 키워드 기반으로 분류
        if has_casual_keyword:
            updated_state = chat_state.update(
                guardrail_passed=True,
                guardrail_confidence=0.7,
                intent_type="casual",
                intent_confidence=0.7,
                intent_analysis_details={"reason": "에러 발생 후 키워드 기반 폴백 분류"},
                intent_category=None,
                blocked=False,
                route=chat_state.route + " -> unified_analysis_error_fallback_casual",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            return to_state_dict(updated_state)
        elif has_business_keyword:
            updated_state = chat_state.update(
                guardrail_passed=True,
                guardrail_confidence=0.7,
                intent_type="business",
                intent_confidence=0.7,
                intent_analysis_details={"reason": "에러 발생 후 키워드 기반 폴백 분류"},
                intent_category="general",
                blocked=False,
                route=chat_state.route + " -> unified_analysis_error_fallback_business",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            return to_state_dict(updated_state)
        else:
            # 키워드도 없으면 기본적으로 casual로 처리 (일상 대화 가능성 높음)
            logger.warning(f"에러 발생 및 키워드 없음: '{user_message[:50]}...' -> casual (기본값)")
            updated_state = chat_state.update(
                guardrail_passed=True,
                guardrail_confidence=0.5,
                intent_type="casual",
                intent_confidence=0.5,
                intent_analysis_details={"reason": "에러 발생 및 키워드 없음, 기본값으로 casual 처리"},
                intent_category=None,
                blocked=False,
                route=chat_state.route + " -> unified_analysis_error_default_casual",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            return to_state_dict(updated_state)

