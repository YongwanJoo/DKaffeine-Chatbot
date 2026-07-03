"""LLM Provider 포트 인터페이스"""
from abc import ABC, abstractmethod
from typing import Optional, List, Any


class LLMPort(ABC):
    """LLM Provider 포트 인터페이스
    
    LLM 및 임베딩 모델 접근을 추상화하는 포트입니다.
    도메인 레이어는 이 인터페이스를 통해 LLM에 접근합니다.
    """
    
    @abstractmethod
    def get_chat_llm(
        self,
        role: str = "business",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Any:
        """역할에 맞는 대화형 LLM 반환
        
        Args:
            role: 'business' | 'casual' (기본값: "business")
            model_name: 모델명 (None이면 role에 따라 자동 선택)
            temperature: 온도 (None이면 role에 따라 자동 설정)
            max_tokens: 최대 토큰 수 (선택적)
        
        Returns:
            LangChain LLM 인스턴스
        """
        pass
    
    @abstractmethod
    def get_embeddings(self) -> Any:
        """임베딩 모델 반환
        
        Returns:
            LangChain Embeddings 인스턴스
        """
        pass
    
    @abstractmethod
    def invoke_with_resilience(self, llm: Any, messages: List[Any]) -> Any:
        """Circuit Breaker를 통해 LLM 호출 (장애 복구)
        
        Args:
            llm: LangChain LLM 인스턴스
            messages: LLM에 전달할 메시지 리스트
        
        Returns:
            LLM 응답 객체
        """
        pass

