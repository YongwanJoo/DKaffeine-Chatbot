"""도메인 예외 클래스

커스텀 예외 계층 구조를 정의하여 일관된 에러 처리를 제공합니다.
"""
from typing import Optional, Dict, Any


class ChatbotException(Exception):
    """챗봇 기본 예외 클래스
    
    모든 챗봇 관련 예외의 기본 클래스입니다.
    """
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            message: 에러 메시지
            status_code: HTTP 상태 코드 (기본: 500)
            error_code: 에러 코드 (선택적)
            details: 추가 상세 정보 (선택적)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class GuardrailException(ChatbotException):
    """Guardrail 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 403,
        error_code: str = "GUARDRAIL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)


class BlacklistException(GuardrailException):
    """Blacklist 차단 예외"""
    def __init__(
        self,
        message: str = "부적절한 내용이 감지되었습니다.",
        category: Optional[str] = None,
        matched_patterns: Optional[list] = None
    ):
        details = {}
        if category:
            details["category"] = category
        if matched_patterns:
            details["matched_patterns"] = matched_patterns
        
        super().__init__(
            message=message,
            status_code=403,
            error_code="BLACKLIST_BLOCKED",
            details=details
        )


class RAGException(ChatbotException):
    """RAG 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "RAG_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)


class RAGSearchException(RAGException):
    """RAG 검색 실패 예외"""
    def __init__(
        self,
        message: str = "문서 검색 중 오류가 발생했습니다.",
        query: Optional[str] = None
    ):
        details = {}
        if query:
            details["query"] = query
        
        super().__init__(
            message=message,
            status_code=500,
            error_code="RAG_SEARCH_ERROR",
            details=details
        )


class RAGAnswerException(RAGException):
    """RAG 답변 생성 실패 예외"""
    def __init__(
        self,
        message: str = "답변 생성 중 오류가 발생했습니다.",
        confidence: Optional[float] = None
    ):
        details = {}
        if confidence is not None:
            details["confidence"] = confidence
        
        super().__init__(
            message=message,
            status_code=500,
            error_code="RAG_ANSWER_ERROR",
            details=details
        )


class FAQException(ChatbotException):
    """FAQ 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "FAQ_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)


class FAQSearchException(FAQException):
    """FAQ 검색 실패 예외"""
    def __init__(
        self,
        message: str = "FAQ 검색 중 오류가 발생했습니다.",
        company_id: Optional[str] = None
    ):
        details = {}
        if company_id:
            details["company_id"] = company_id
        
        super().__init__(
            message=message,
            status_code=500,
            error_code="FAQ_SEARCH_ERROR",
            details=details
        )


class CacheException(ChatbotException):
    """Cache 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "CACHE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)


class LLMException(ChatbotException):
    """LLM 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "LLM_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)


class LLMTimeoutException(LLMException):
    """LLM 타임아웃 예외"""
    def __init__(
        self,
        message: str = "LLM 응답 시간이 초과되었습니다.",
        timeout: Optional[float] = None
    ):
        details = {}
        if timeout is not None:
            details["timeout"] = timeout
        
        super().__init__(
            message=message,
            status_code=504,
            error_code="LLM_TIMEOUT",
            details=details
        )


class ConfigurationException(ChatbotException):
    """설정 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "CONFIG_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)


class ValidationException(ChatbotException):
    """검증 관련 예외"""
    def __init__(
        self,
        message: str,
        status_code: int = 422,
        error_code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, error_code, details)

