"""채팅 히스토리 검증 서비스 (보안 강화)"""
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ChatHistoryValidator:
    """채팅 히스토리 검증 및 정규화
    
    토큰 기반 제한 고려, 보수적인 길이 제한 적용
    """
    
    MAX_HISTORY_LENGTH = 6  # 최대 3턴 (user-assistant 쌍)
    # 보수적인 길이 제한 (한글 4000자 기준, 토큰 약 2000개)
    # 한글 1자 ≈ 0.5 토큰 (Claude 모델 기준)
    MAX_MESSAGE_LENGTH = 4000  # 개별 메시지 최대 길이 (한글 4000자)
    MAX_TOTAL_LENGTH = 8000  # 전체 히스토리 최대 길이 (한글 8000자, 토큰 약 4000개)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """토큰 수 추정 (대략적)
        
        단순 글자 수가 아닌 토큰 기반 제한 고려
        - 영어/숫자/기호: 1자 ≈ 0.25 토큰
        - 한글: 1자 ≈ 0.5 토큰
        - 공백: 1자 ≈ 0.1 토큰
        
        Args:
            text: 토큰 수를 추정할 텍스트
            
        Returns:
            추정 토큰 수
        """
        if not text:
            return 0
        
        # 한글 문자 수
        korean_count = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        # 영어/숫자/기호 문자 수
        other_count = len(text) - korean_count - text.count(' ')
        # 공백 수
        space_count = text.count(' ')
        
        # 토큰 수 추정
        estimated_tokens = (
            korean_count * 0.5 +  # 한글: 1자 ≈ 0.5 토큰
            other_count * 0.25 +  # 영어/숫자/기호: 1자 ≈ 0.25 토큰
            space_count * 0.1      # 공백: 1자 ≈ 0.1 토큰
        )
        
        return int(estimated_tokens)
    
    @classmethod
    def validate_and_normalize(
        cls,
        chat_history: Optional[List[Dict[str, Any]]],
        validate_content: bool = True
    ) -> List[Dict[str, str]]:
        """히스토리 검증 및 정규화
        
        Args:
            chat_history: 검증할 히스토리
            validate_content: 내용 검증 여부 (가드레일 적용)
            
        Returns:
            정규화된 히스토리 리스트
        """
        if not chat_history:
            return []
        
        if not isinstance(chat_history, list):
            logger.warning(f"Invalid chat_history type: {type(chat_history)}, expected list")
            return []
        
        # 길이 제한
        if len(chat_history) > cls.MAX_HISTORY_LENGTH:
            logger.warning(f"Chat history too long: {len(chat_history)}, truncating to {cls.MAX_HISTORY_LENGTH}")
            chat_history = chat_history[-cls.MAX_HISTORY_LENGTH:]
        
        normalized = []
        total_length = 0
        
        for idx, msg in enumerate(chat_history):
            if not isinstance(msg, dict):
                logger.warning(f"Invalid message type at index {idx}: {type(msg)}, skipping")
                continue
            
            # 필수 필드 검증
            role = msg.get("role")
            content = msg.get("content")
            
            if not role or not content:
                logger.warning(f"Missing required fields at index {idx}: role={role}, content={bool(content)}")
                continue
            
            # role 검증
            if role not in ["user", "assistant"]:
                logger.warning(f"Invalid role at index {idx}: {role}, skipping")
                continue
            
            # content 타입 검증
            if not isinstance(content, str):
                logger.warning(f"Invalid content type at index {idx}: {type(content)}, converting to string")
                content = str(content)
            
            # 토큰 기반 길이 제한 고려
            # 메시지 길이 제한 (문자 수)
            if len(content) > cls.MAX_MESSAGE_LENGTH:
                logger.warning(f"Message too long at index {idx}: {len(content)} chars, truncating to {cls.MAX_MESSAGE_LENGTH}")
                content = content[:cls.MAX_MESSAGE_LENGTH]
            
            # 토큰 수 추정
            estimated_tokens = cls._estimate_tokens(content)
            max_tokens_per_message = cls._estimate_tokens("가" * cls.MAX_MESSAGE_LENGTH)  # 약 2000 토큰
            
            if estimated_tokens > max_tokens_per_message:
                # 토큰 수가 초과하면 추가로 자르기
                # 대략적으로 토큰 수에 비례하여 자르기
                ratio = max_tokens_per_message / estimated_tokens
                new_length = int(len(content) * ratio * 0.9)  # 10% 여유
                content = content[:new_length]
                logger.warning(f"Message tokens exceeded at index {idx}: {estimated_tokens} tokens, truncating to {new_length} chars")
            
            # 전체 길이 체크 (문자 수)
            total_length += len(content)
            if total_length > cls.MAX_TOTAL_LENGTH:
                logger.warning(f"Total history length exceeded: {total_length} chars, stopping")
                break
            
            # 정규화된 메시지 추가
            normalized.append({
                "role": role,
                "content": content
            })
        
        logger.debug(f"Chat history validated: {len(normalized)} messages, total length: {total_length}")
        return normalized
    
    @classmethod
    def validate_content(
        cls,
        content: str,
        guardrail_service: Optional[Any] = None
    ) -> tuple[bool, Optional[str]]:
        """메시지 내용 검증 (가드레일 적용)
        
        Args:
            content: 검증할 메시지 내용
            guardrail_service: GuardrailService 인스턴스 (선택)
            
        Returns:
            (is_valid, reason) 튜플
        """
        if not guardrail_service:
            # 기본 검증만 수행
            if not content or not content.strip():
                return False, "Empty message"
            if len(content) > cls.MAX_MESSAGE_LENGTH:
                return False, f"Message too long: {len(content)}"
            return True, None
        
        # Guardrail 검증
        try:
            result = guardrail_service.validate(content)
            if result.blocked:
                return False, result.reason
            return True, None
        except Exception as e:
            logger.error(f"Guardrail validation error: {e}")
            # 검증 실패 시 기본 검증만 수행
            return True, None
    
    @classmethod
    def sanitize_history(
        cls,
        chat_history: Optional[List[Dict[str, Any]]],
        guardrail_service: Optional[Any] = None,
        trusted_source: bool = True
    ) -> List[Dict[str, str]]:
        """히스토리 검증 및 정제 (가드레일 적용)
        
        Args:
            chat_history: 검증할 히스토리
            guardrail_service: GuardrailService 인스턴스 (선택)
            trusted_source: 히스토리 출처가 신뢰할 수 있는지 여부
                           (True: Redis 등 서버에서 생성, False: 클라이언트 제공)
            
        Returns:
            정제된 히스토리 리스트 (차단된 메시지는 제외)
        """
        normalized = cls.validate_and_normalize(chat_history, validate_content=False)
        
        if not guardrail_service:
            # 가드레일 서비스가 없으면 기본 검증만 수행
            # 단, 신뢰할 수 없는 출처의 assistant 메시지는 제거
            if not trusted_source:
                sanitized = [msg for msg in normalized if msg["role"] == "user"]
                logger.warning(
                    f"보안: 신뢰할 수 없는 출처의 히스토리에서 assistant 메시지 제거: "
                    f"{len(normalized)} -> {len(sanitized)}"
                )
                return sanitized
            return normalized
        
        # 각 메시지 내용 검증
        sanitized = []
        for msg in normalized:
            role = msg["role"]
            content = msg["content"]
            
            # 보안 강화: 신뢰할 수 없는 출처의 assistant 메시지는 절대 신뢰하지 않음
            if not trusted_source and role == "assistant":
                logger.warning(
                    f"보안: 신뢰할 수 없는 출처의 assistant 메시지 제거: "
                    f"content='{content[:50]}...'"
                )
                continue
            
            # 사용자 메시지는 항상 가드레일 검증
            # 단, trusted_source(Redis)인 경우 이미 검증된 것으로 간주하여 스킵 (성능 최적화)
            if role == "user":
                is_valid, reason = cls.validate_content(content, guardrail_service)
                if not is_valid:
                    logger.warning(f"Blocked message in history: {reason}, skipping")
                    continue
            
            # 보안 강화: assistant 메시지도 최소한 블랙리스트 검증 수행
            # LLM 출력도 신뢰할 수 없으므로 (문서 오염, LLM 탈옥, 프롬프트 인젝션 등)
            # 2차 프롬프트 인젝션 공격을 방지하기 위해 검증 필요
            # 단, trusted_source(Redis)인 경우 이미 검증된 것으로 간주하여 스킵 (성능 최적화)
            if role == "assistant" and guardrail_service and not trusted_source:
                # 최소한 블랙리스트 검증 (빠르고 효과적)
                # LLM 기반 검증은 비용이 들 수 있으므로 블랙리스트만 수행
                try:
                    blacklist_result = guardrail_service.check_blacklist(content)
                    if blacklist_result.blocked:
                        logger.warning(
                            f"보안: 히스토리의 assistant 메시지가 블랙리스트에 차단되어 제거: "
                            f"category={blacklist_result.category}, reason={blacklist_result.reason}, "
                            f"content='{content[:50]}...'"
                        )
                        continue  # 차단된 메시지는 히스토리에서 제거
                except Exception as e:
                    # 검증 실패 시 로그만 남기고 통과 (안정성 우선)
                    logger.warning(f"Assistant 메시지 검증 중 오류 (통과): {e}")
            
            sanitized.append(msg)
        
        logger.info(f"History sanitized: {len(normalized)} -> {len(sanitized)} messages")
        return sanitized

