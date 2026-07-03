"""Guardrail 서비스 포트 (인터페이스)"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, List


class GuardrailResult:
    """Guardrail 검증 결과"""
    def __init__(
        self,
        blocked: bool,
        reason: Optional[str] = None,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        matched_patterns: Optional[List[str]] = None,
        details: Optional[Dict] = None
    ):
        self.blocked = blocked
        self.reason = reason
        self.category = category
        self.confidence = confidence
        self.matched_patterns = matched_patterns or []
        self.details = details or {}


class GuardrailPort(ABC):
    """Guardrail 서비스 포트 (인터페이스)
    
    Guardrail 검증 서비스의 추상 인터페이스입니다.
    모든 Guardrail 서비스 구현체는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def check_blacklist(self, text: str) -> GuardrailResult:
        """Blacklist 기반 검증 (1차, 가장 빠름)
        
        Args:
            text: 검증할 텍스트
            
        Returns:
            GuardrailResult: 검증 결과
        """
        pass
    
    @abstractmethod
    def check_with_llm(
        self,
        text: str,
        context: Optional[Dict] = None,
        chat_history: Optional[list] = None,
        chatbot_settings: Optional[Dict] = None
    ) -> GuardrailResult:
        """LLM 기반 검증 (2차, 가장 정교함)
        
        Args:
            text: 검증할 텍스트
            context: 추가 컨텍스트 (선택적)
            chat_history: 대화 히스토리 (선택적)
            chatbot_settings: 챗봇 설정 (선택적)
            
        Returns:
            GuardrailResult: 검증 결과
        """
        pass
    
    @abstractmethod
    def validate(self, text: str, user_id: Optional[str] = None) -> GuardrailResult:
        """통합 검증 (모든 단계 수행)
        
        Args:
            text: 검증할 텍스트
            user_id: 사용자 ID (선택적)
            
        Returns:
            GuardrailResult: 검증 결과
        """
        pass

