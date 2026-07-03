"""LLM Provider 어댑터"""
from typing import Optional, List, Any

from app.domain.ports.llm_port import LLMPort
from app.infrastructure.adapters.llm import LLMProvider


class LLMAdapter(LLMPort):
    """LLM Provider 어댑터 구현
    
    LLMPort 인터페이스를 구현하여 LLM 접근을 제공합니다.
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """LLM Provider 어댑터 초기화
        
        Args:
            llm_provider: LLMProvider 인스턴스 (None이면 새로 생성)
        """
        self._provider = llm_provider or LLMProvider()
    
    def get_chat_llm(
        self,
        role: str = "business",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Any:
        """역할에 맞는 대화형 LLM 반환"""
        return self._provider.get_chat_llm(role, model_name, temperature, max_tokens)
    
    def get_embeddings(self) -> Any:
        """임베딩 모델 반환"""
        return self._provider.get_embeddings()
    
    def invoke_with_resilience(self, llm: Any, messages: List[Any]) -> Any:
        """Circuit Breaker를 통해 LLM 호출"""
        return self._provider.invoke_with_resilience(llm, messages)

    async def ainvoke_with_resilience(self, llm: Any, messages: List[Any]) -> Any:
        """Circuit Breaker를 통해 LLM 비동기 호출"""
        return await self._provider.ainvoke_with_resilience(llm, messages)

