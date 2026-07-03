"""
강화된 Guardrail 서비스
- Blacklist 기반 차단
- LLM 기반 검증
- 상세한 로깅 및 모니터링
"""
import re
import unicodedata
import json
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import LangChainException

from app.domain.ports.guardrail_port import GuardrailPort, GuardrailResult
from app.domain.ports.llm_port import LLMPort
from app.infrastructure.utils.resilience import CircuitBreaker
from app.infrastructure.config.config_loader import get_config_int, get_config_float

# 로깅 설정
logger = logging.getLogger(__name__)

class GuardrailService(GuardrailPort):
    """강화된 Guardrail 서비스
    
    부적절한 콘텐츠를 차단하는 다층 방어 시스템입니다.
    
    검증 단계:
    1. **Blacklist 체크**: 패턴 기반 1차 필터링 (가장 빠름)
    2. **LLM 검증**: LLM 기반 2차 검증 (가장 정교함)
    
    특징:
    - 다층 방어 체계
    - 화이트리스트 지원
    - 텍스트 정규화 (우회 시도 방지)
    - 변형 패턴 감지
    
    Example:
        ```python
        service = GuardrailService()
        result = service.validate("사용자 질문", user_id="user_001")
        
        if result.blocked:
            print(f"차단됨: {result.reason}")
        ```
    
    Args:
        blacklist_path: Blacklist JSON 파일 경로
        config_path: Guardrail 설정 JSON 파일 경로
    """
    
    def __init__(
        self,
        blacklist_path: str = "./config/blacklist.json",
        config_path: str = "./config/guardrail_config.json",
        llm_provider: Optional[LLMPort] = None,
        redis_client: Optional[Any] = None
    ):
        self.blacklist_path = Path(blacklist_path)
        self.config_path = Path(config_path)
        
        # 설정 로드
        self.config = self._load_config()
        
        # 데이터 로드
        self.blacklist = self._load_blacklist()
        
        # LLM 초기화 (설정 기반, 지연 초기화)
        self.llm_config = self.config.get("check_levels", {}).get("llm_check", {})
        self.llm_enabled = self.config.get("enabled", True) and self.llm_config.get("enabled", True)
        self.llm = None  # 지연 초기화
        # LLM Provider 주입 (없으면 기본 생성)
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            # 하위 호환성: 직접 생성
            from app.infrastructure.adapters.llm.llm_adapter import LLMAdapter
            self.llm_provider = LLMAdapter()
        
        # Circuit Breaker 초기화 (LLM API 장애 복구)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=get_config_int("GUARDRAIL_LLM_CIRCUIT_BREAKER_THRESHOLD", 5, section="guardrail"),
            recovery_timeout=get_config_float("GUARDRAIL_LLM_CIRCUIT_BREAKER_TIMEOUT", 60.0, section="guardrail"),
            expected_exception=(LangChainException, Exception),
            name="guardrail_llm",
            redis_client=redis_client
        )
        
        # 화이트리스트
        self.whitelist = self.config.get("whitelist_patterns", [])
        
        logger.info("GuardrailService initialized")
        logger.info(f"Blacklist categories: {len(self.blacklist)}")
        logger.info(
            f"Guardrail LLM CircuitBreaker 초기화: "
            f"threshold={self.circuit_breaker.failure_threshold}, "
            f"timeout={self.circuit_breaker.recovery_timeout}s"
        )
    
    def _load_config(self) -> dict:
        """Guardrail 설정 로드"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return {
                "enabled": True,
                "strict_mode": True,
                "max_query_length": 1000,
                "min_query_length": 2,
                "whitelist_patterns": []
            }
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {"enabled": True, "strict_mode": True}
    
    def _load_blacklist(self) -> dict:
        """Blacklist 로드"""
        if not self.blacklist_path.exists():
            logger.warning(f"Blacklist file not found: {self.blacklist_path}")
            return {}
        
        try:
            with open(self.blacklist_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading blacklist: {e}")
            return {}
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 유니코드 정규화
        normalized = unicodedata.normalize("NFKC", text)
        # 소문자 변환
        normalized = normalized.lower()
        # 공백 정규화
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
    
    def _check_whitelist(self, text: str) -> bool:
        """화이트리스트 체크 (화이트리스트에 있으면 자동 통과)"""
        normalized = self._normalize_text(text)
        
        for pattern in self.whitelist:
            if pattern.lower() in normalized:
                logger.debug(f"Whitelist match: {pattern}")
                return True
        return False
    
    def check_length(self, text: str) -> Tuple[bool, Optional[str]]:
        """질문 길이 검증"""
        max_len = self.config.get("max_query_length", 1000)
        min_len = self.config.get("min_query_length", 2)
        
        if len(text) > max_len:
            return False, f"질문이 너무 깁니다 (최대 {max_len}자)"
        if len(text) < min_len:
            return False, f"질문이 너무 짧습니다 (최소 {min_len}자)"
        
        return True, None
    
    def check_blacklist(self, text: str) -> GuardrailResult:
        """Blacklist 기반 검증 (1차, 가장 빠름)"""
        if not self.config.get("enabled", True):
            return GuardrailResult(blocked=False)
        
        # 화이트리스트 체크
        if self._check_whitelist(text):
            return GuardrailResult(blocked=False, details={"whitelisted": True})
        
        normalized = self._normalize_text(text)
        # 공백 제거 버전도 체크 (우회 시도 방지)
        normalized_no_space = re.sub(r"\s+", "", normalized)
        
        matched_patterns = []
        blocked_categories = []
        
        for category, data in self.blacklist.items():
            if not isinstance(data, dict):
                continue
            
            is_strict = data.get("strict", True)
            patterns = data.get("patterns", [])
            variations = data.get("variations", [])
            
            # 직접 패턴 체크
            for pattern in patterns:
                pattern_lower = pattern.lower()
                # 정규화된 텍스트에서 검색
                if pattern_lower in normalized or pattern_lower in normalized_no_space:
                    matched_patterns.append(pattern)
                    blocked_categories.append(category)
                    logger.warning(f"Blacklist match: category={category}, pattern={pattern}")
                    break
            
            # 변형 패턴 체크
            for variation in variations:
                base = variation.get("base", "").lower()
                variants = [v.lower() for v in variation.get("variants", [])]
                
                # base는 그룹화 키일 뿐, 단독 매칭하지 않음 (variants만 매칭)
                # 예: "전화번호"는 매칭하지 않고, "전화번호알려줘"만 매칭
                for pat in variants:
                    if pat in normalized or pat in normalized_no_space:
                        matched_patterns.append(f"{base} (variant: {pat})")
                        blocked_categories.append(category)
                        logger.warning(f"Blacklist variant match: category={category}, variant={pat}")
                        break
        
        if matched_patterns:
            # 가장 심각한 카테고리 찾기 (strict 우선)
            primary_category = None
            for cat in blocked_categories:
                if self.blacklist.get(cat, {}).get("strict", True):
                    primary_category = cat
                    break
            if not primary_category:
                primary_category = blocked_categories[0]
            
            messages = self.config.get("response_messages", {})
            reason = messages.get(f"blocked_{primary_category}", messages.get("blocked_general", "부적절한 내용이 포함되어 있습니다."))
            
            return GuardrailResult(
                blocked=True,
                reason=reason,
                category=primary_category,
                matched_patterns=matched_patterns,
                details={
                    "categories": blocked_categories,
                    "match_count": len(matched_patterns)
                }
            )
        
        return GuardrailResult(blocked=False)
    
    def _ensure_llm_initialized(self):
        """LLM 초기화 (지연 초기화)"""
        if self.llm is None and self.llm_enabled:
            try:
                self.llm = self.llm_provider.get_chat_llm(
                    role="casual",
                    temperature=self.llm_config.get("temperature", 0),
                    max_tokens=None
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for guardrail: {e}")
                self.llm = None
    
    def check_with_llm(self, text: str, context: Optional[Dict] = None, chat_history: Optional[list] = None, chatbot_settings: Optional[Dict] = None) -> GuardrailResult:
        """LLM 기반 검증 (3차, 가장 정교함)
        
        Args:
            text: 검증할 텍스트
            context: 추가 컨텍스트 정보
            chat_history: 대화 히스토리 (봇이 먼저 일상 대화를 시작한 경우 고려)
            chatbot_settings: 챗봇 설정 (키워드 및 임계값 포함, 선택적 - 파일 시스템 설정 우선)
        """
        if not self.config.get("enabled", True):
            return GuardrailResult(blocked=False)
        
        # 이전 검증에서 차단되었으면 LLM 검증 생략
        if context and context.get("blacklist_blocked"):
            return GuardrailResult(
                blocked=True,
                reason="이전 검증에서 차단됨",
                details={"skipped_llm": True}
            )
        
        # 대화 맥락 확인: 봇이 먼저 일상 대화를 시작한 경우, 사용자 응답에 대해 가드레일 완화
        # 보안: chat_history는 이미 messages.py에서 검증되어 Redis에서 가져온 신뢰할 수 있는 히스토리만 포함
        # 보안: 클라이언트 제공 히스토리는 이미 무시되었고, assistant 메시지도 신뢰할 수 없는 출처에서는 제거됨
        if chat_history:
            logger.debug(f"가드레일: 히스토리 확인 중, 히스토리 길이={len(chat_history)}")
            
            # 최근 메시지가 어시스턴트(봇)에서 시작했는지 확인
            # 보안: 이 히스토리는 서버(Redis)에서 생성된 것이므로 신뢰 가능
            last_message = chat_history[-1]
            last_role = ""
            last_content = ""
            
            if isinstance(last_message, dict):
                last_role = last_message.get("role", "")
                last_content = last_message.get("content", "")
                logger.debug(f"가드레일: 마지막 메시지 role={last_role}, content='{last_content[:50]}...'")
            else:
                # LangChain 메시지 객체인 경우
                from langchain_core.messages import BaseMessage, AIMessage
                if isinstance(last_message, BaseMessage):
                    last_role = "assistant" if isinstance(last_message, AIMessage) else "user"
                    last_content = getattr(last_message, "content", "")
                    logger.debug(f"가드레일: LangChain 메시지, role={last_role}, content='{last_content[:50]}...'")
            
            # 봇이 먼저 일상 대화를 시작한 경우 (음식 추천, 날씨 등)
            # 보안: 이 assistant 메시지는 서버에서 생성된 것이므로 신뢰 가능
            if last_role == "assistant" and last_content:
                # 파일 시스템 설정에서 키워드 가져오기 (우선순위: 파일 시스템 > DB 설정 > 기본값)
                keywords_config = self.config.get("guardrail_keywords", {})
                if not keywords_config and chatbot_settings:
                    keywords_config = chatbot_settings.get("guardrail_keywords", {})
                # guardrail_config.json에서 로드 (하드코딩 제거)
                casual_keywords = keywords_config.get("bot_initiated_casual_keywords", [])
                matched_keyword = None
                for keyword in casual_keywords:
                    if keyword in last_content:
                        matched_keyword = keyword
                        break
                
                if matched_keyword:
                    logger.info(
                        f"✅ 가드레일 완화: 봇이 먼저 일상 대화를 시작함 (키워드='{matched_keyword}') "
                        f"봇 메시지='{last_content[:50]}...', 사용자 메시지='{text[:50]}...'"
                    )
                    # 가드레일을 완화하여 통과시킴
                    return GuardrailResult(
                        blocked=False,
                        reason="봇이 시작한 일상 대화에 대한 응답",
                        category="allowed",
                        confidence=0.9,
                        details={"context_aware": True, "bot_initiated_casual": True, "matched_keyword": matched_keyword}
                    )
                else:
                    logger.debug(f"가드레일: 봇 메시지에 일상 대화 키워드 없음 - '{last_content[:50]}...'")
            else:
                logger.debug(f"가드레일: 마지막 메시지가 봇이 아님 (role={last_role})")
        else:
            logger.debug("가드레일: 히스토리가 비어있음")
        
        # 키워드 기반 차단 체크 (LLM 호출 전에 먼저 실행) - 우선순위 최우선
        text_lower = text.lower()
        # 단어 단위로 분리 (공백, 구두점 기준)
        import re
        words = set(re.findall(r'\b\w+\b', text_lower))  # 단어 경계 기준으로 추출
        
        # 파일 시스템 설정에서 키워드 가져오기 (우선순위: 파일 시스템 > DB 설정)
        # guardrail_config.json에서 로드 (하드코딩 제거)
        keywords_config = self.config.get("guardrail_keywords", {})
        if not keywords_config and chatbot_settings:
            keywords_config = chatbot_settings.get("guardrail_keywords", {})
        blocked_topic_keywords = keywords_config.get("blocked_topic_keywords", [])
        # 단어 단위로 정확히 매칭 (부분 문자열 매칭 방지)
        has_blocked_keyword = any(kw in words for kw in blocked_topic_keywords)
        
        # 차단 키워드가 있으면 무조건 차단 (LLM 호출 없이 즉시 반환)
        if has_blocked_keyword:
            matched_keywords = [kw for kw in blocked_topic_keywords if kw in words]
            logger.warning(f"LLM Guardrail: blocked topic keyword detected: {matched_keywords}")
            return GuardrailResult(
                blocked=True,
                reason="업무와 무관한 주제입니다",
                category="off_topic",
                confidence=0.95,
                details={"blocked_keyword": True, "matched_keywords": matched_keywords}
            )
        
        check_config = self.config.get("check_levels", {}).get("llm_check", {})
        if not check_config.get("enabled", True):
            return GuardrailResult(blocked=False)
        
        # LLM 초기화 (지연 초기화)
        self._ensure_llm_initialized()
        if self.llm is None:
            return GuardrailResult(blocked=False)
        
        system_prompt = """당신은 B2B 업무용 챗봇의 안전 필터입니다.
**중요: 다음 일상 대화 주제는 항상 허용하세요 (is_valid=true, category=allowed):**
- 맛집/음식 추천 (주변 맛집, 맛있는 곳, 식당 추천 등)
- 점심/저녁 추천 (점심 뭐 먹을까, 저녁 메뉴 추천, 식사 메뉴 등)
- 건강 관련 (건강 관리, 운동, 건강한 생활 등)
- 일상적인 인사말 (안녕하세요, 감사합니다 등)

**항상 허용 (is_valid=true, business_related=true):**
- 위의 일상 대화 주제들
- **업무 관련 키워드가 포함된 모든 질문** (휴가, 연차, 신청, 취소, 방법, 절차, 출장, 복지, 복장, 시스템, 도구 등)
- 업무, 회사 규정, 절차 관련 질문
- 사내 시스템 사용법, 도구 사용법
- 일반적인 비즈니스 지식 질문
- **회사/조직/부서/센터/팀 관련 질문** (예: "카카오 i 커넥트 센터", "XX 부서", "XX 팀", "XX 센터에 대해 알려줘" 등)
- **회사 정보 관련 질문** (회사 구성, 조직 구조, 인력 구성, 개발자 구성, 팀 구성, 부서 구성 등)
- **정보 요청 질문** ("~에 대해 알려줘", "~이 뭐야", "~에 대해 설명해줘", "~는 어떻게 되나요" 등은 회사/조직 관련이면 업무 질문으로 판단)
- **특히 주의: "휴가", "연차", "신청", "취소", "방법", "절차", "센터", "부서", "팀", "조직", "구성", "구조", "인력", "개발자" 등의 키워드가 포함되면 반드시 업무 질문으로 판단하세요**

**반드시 차단 (is_valid=false, category=off_topic):**
- 정치 관련 주제 (정치인 이름, 정당, 선거, 정치적 의견, 정치 뉴스 등)
- 종교 관련 주제 (종교 교리, 종교인, 종교적 의견 등)
- 업무와 전혀 무관하고 위의 허용 주제에도 해당하지 않는 사적인 대화
  (예: 연애 상담, 개인적인 고민, 게임, 유명인 사생활 등)

**반드시 차단 (is_valid=false):**
- 선정적/음란한 내용 (category: sexual)
- 욕설, 비속어, 혐오 표현 (category: profanity)
- 폭력적이거나 위협적인 내용 (category: violence)
- 개인정보 노출 시도 (category: personal_info)
- 스팸/광고성 내용 (category: spam)

**응답 형식 (반드시 JSON):**
{
  "is_valid": true/false,
  "reason": "판단 사유 (한국어)",
  "confidence": 0.0-1.0,
  "category": "allowed|profanity|sexual|violence|spam|personal_info|off_topic",
  "details": {
    "business_related": true/false,
    "inappropriate_content": true/false
  }
}"""
        
        user_prompt = f"다음 사용자 질문을 검증하세요:\n\n{text}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Circuit Breaker를 통해 LLM 호출 (장애 복구)
            def _invoke():
                return self.llm.invoke(messages)
            response = self.circuit_breaker.call(_invoke)
            content = response.content.strip()
            
            # JSON 추출 (마크다운 코드블록 제거)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            is_valid = result.get("is_valid", False)
            reason = result.get("reason", "Unknown")
            confidence = result.get("confidence", 0.5)
            category = result.get("category", "unknown")
            details = result.get("details", {})
            
            # 업무 관련 키워드가 포함된 경우 무조건 허용 (LLM 판단보다 우선)
            # 파일 시스템 설정에서 키워드 가져오기 (우선순위: 파일 시스템 > DB 설정)
            # guardrail_config.json에서 로드 (하드코딩 제거)
            keywords_config = self.config.get("guardrail_keywords", {})
            if not keywords_config and chatbot_settings:
                keywords_config = chatbot_settings.get("guardrail_keywords", {})
            business_keywords = keywords_config.get("business_keywords", [])
            has_business_keyword = any(kw in text_lower for kw in business_keywords)
            
            if has_business_keyword:
                # 업무 키워드가 있으면 무조건 허용 (LLM이 off_topic으로 분류해도 오버라이드)
                logger.info(f"LLM Guardrail: business keyword detected, overriding LLM decision (category={category})")
                return GuardrailResult(blocked=False, reason="업무 관련 질문으로 판단됨", category="allowed")
            
            # 허용된 일상 대화 주제 키워드 (맛집, 점심/저녁, 건강, 인사말)
            # guardrail_config.json에서 로드 (하드코딩 제거)
            allowed_casual_keywords = keywords_config.get("casual_keywords", [])
            has_casual_keyword = any(kw in text_lower for kw in allowed_casual_keywords)
            
            # 허용된 일상 대화 주제가 포함된 경우 허용 (부적절한 카테고리가 아닌 경우)
            if has_casual_keyword and category not in ["profanity", "sexual", "violence", "spam", "personal_info", "off_topic"]:
                logger.info(f"LLM Guardrail: allowed casual topic detected (keyword match), allowing")
                return GuardrailResult(blocked=False, reason="allowed casual topic")
            
            # "off_topic"이지만 허용된 키워드가 있는 경우는 위에서 처리됨
            # "off_topic"이고 허용 키워드가 없는 경우는 차단 (업무와 무관한 사적인 대화)
            
            if not is_valid:
                messages_config = self.config.get("response_messages", {})
                # 카테고리 매핑 (LLM 응답 카테고리 → 설정 파일 키)
                category_map = {
                    "profanity": "blocked_profanity",
                    "sexual": "blocked_sexual",
                    "violence": "blocked_violence",
                    "personal_info": "blocked_personal_info",
                    "spam": "blocked_spam",
                    # "off_topic"은 제거 - 일상 대화는 허용해야 함
                }
                config_key = category_map.get(category, "blocked_general")
                user_facing_reason = messages_config.get(config_key, messages_config.get("blocked_general", reason))
                
                logger.warning(f"LLM Guardrail blocked: category={category}, reason={reason}, confidence={confidence}")
                
                return GuardrailResult(
                    blocked=True,
                    reason=user_facing_reason,
                    category=category,
                    confidence=confidence,
                    details=details
                )
            
            logger.debug(f"LLM Guardrail passed: confidence={confidence}")
            return GuardrailResult(
                blocked=False,
                confidence=confidence,
                details=details
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM response JSON parse error: {e}, content: {content[:100]}")
            # JSON 파싱 실패 시 보수적으로 차단
            return GuardrailResult(
                blocked=True,
                reason="검증 과정에서 오류가 발생했습니다.",
                details={"error": "json_parse_error"}
            )
        except Exception as e:
            logger.error(f"LLM Guardrail error: {e}")
            # LLM 실패 시 보수적으로 허용 (strict_mode가 아닌 경우)
            if self.config.get("strict_mode", True):
                return GuardrailResult(
                    blocked=True,
                    reason="검증 시스템 오류로 인해 차단되었습니다.",
                    details={"error": str(e)}
                )
            return GuardrailResult(blocked=False, details={"error": str(e)})
    
    def validate(self, text: str, user_id: Optional[str] = None) -> GuardrailResult:
        """통합 검증 (모든 단계 수행)"""
        start_time = datetime.now()
        
        # 길이 체크
        length_ok, length_reason = self.check_length(text)
        if not length_ok:
            return GuardrailResult(
                blocked=True,
                reason=length_reason,
                category="length",
                details={"length": len(text)}
            )
        
        # 1차: Blacklist 체크 (가장 빠름)
        blacklist_result = self.check_blacklist(text)
        if blacklist_result.blocked:
            logger.info(f"Guardrail blocked (blacklist): user={user_id}, category={blacklist_result.category}")
            return blacklist_result
        
        # 2차: LLM 체크
        llm_result = self.check_with_llm(text, context={"blacklist_blocked": False})
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Guardrail validation completed in {elapsed:.2f}s: blocked={llm_result.blocked}")
        
        return llm_result
