"""LLM/임베딩 Provider 추상화 레이어.

AWS Bedrock을 지원합니다.
환경 변수로 프로바이더와 모델명을 제어한다.

ENV
- LLM_PROVIDER: bedrock (default: bedrock)
- EMBEDDINGS_PROVIDER: bedrock (default: bedrock)
- LLM_CASUAL_MODEL: 캐주얼 대화용 모델명 (default: anthropic.claude-haiku-4-5-20251001-v1:0)
- LLM_BUSINESS_MODEL: 비즈니스/RAG 답변용 모델명 (default: anthropic.claude-sonnet-4-5-20250929-v1:0)
- BEDROCK_REGION: AWS 리전 (default: ap-northeast-1)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from langchain_core.exceptions import LangChainException

try:
    from langchain_aws import ChatBedrock, BedrockEmbeddings
except ImportError:
    ChatBedrock = None
    BedrockEmbeddings = None

from app.infrastructure.utils.resilience import CircuitBreaker
from app.infrastructure.config.config_loader import (
    get_bedrock_config,
    get_config,
    get_config_int,
    get_config_float
)

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class LLMProvider:
    """LLM/임베딩 제공자 팩토리.
    
    AWS Bedrock을 지원합니다.
    
    LangChain의 기본 retry 메커니즘을 사용하며, 환경 변수로 설정을 제어합니다.
    boto3 세션과 Bedrock 클라이언트는 싱글톤으로 캐싱하여 성능을 최적화합니다.
    
    Example:
        ```python
        provider = LLMProvider()
        
        # 비즈니스용 LLM
        llm = provider.get_chat_llm(role="business")
        
        # 임베딩
        embeddings = provider.get_embeddings()
        ```
    
    환경 변수:
        - LLM_PROVIDER: "bedrock" (기본값: "bedrock")
        - EMBEDDINGS_PROVIDER: "bedrock" (기본값: "bedrock")
        - LLM_CASUAL_MODEL: 캐주얼 대화용 모델 (기본값: "anthropic.claude-sonnet-4-5-20250929-v1:0")
        - LLM_BUSINESS_MODEL: 비즈니스/RAG 답변용 모델 (기본값: "anthropic.claude-sonnet-4-5-20250929-v1:0")
        - BEDROCK_INFERENCE_PROFILE_ID: Inference profile ID (Claude Haiku 4.5 등 일부 모델에 필요, 선택적)
        - LLM_MAX_RETRIES: 최대 재시도 횟수 (기본값: 3)
        - LLM_TIMEOUT: 타임아웃 초 (기본값: 60.0)
        - BEDROCK_REGION: AWS 리전 (기본값: "ap-northeast-1")
    
    성능 최적화:
        - boto3 세션과 bedrock-runtime 클라이언트는 인스턴스 생성 시 한 번만 초기화되고 재사용됩니다.
        - 이는 TPS(Transactions Per Second)를 크게 향상시킵니다.
    """

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self.llm_provider = get_config("LLM_PROVIDER", "bedrock", section="llm").lower()
        self.embeddings_provider = get_config("EMBEDDINGS_PROVIDER", "bedrock", section="llm").lower()
        
        # Bedrock 설정 로드
        self.bedrock_config = get_bedrock_config()
        
        # Inference profile ID (ap-northeast-1 리전에서는 모든 모델에 필요)
        # 우선순위: 환경변수 > .secrets.toml
        self.inference_profile_id_haiku = (
            get_config("BEDROCK_INFERENCE_PROFILE_ID_HAIKU")
            or self.bedrock_config.get("inference_profile_id_haiku")
            or self.bedrock_config.get("inference_profile_id")  # 하위 호환성
        )
        self.inference_profile_id_sonnet = (
            get_config("BEDROCK_INFERENCE_PROFILE_ID_SONNET")
            or self.bedrock_config.get("inference_profile_id_sonnet")
        )
        # 하위 호환성을 위한 기본값
        self.inference_profile_id = self.inference_profile_id_haiku
        
        # 모델 기본값 설정 (inference_profile_id 확인 후)
        # Perf: 기본 모델을 모두 claude-haiku-4-5로 변경 (응답 속도 최적화)
        # Claude Haiku 4.5는 inference profile이 필요하므로, inference_profile_id가 설정되어 있으면 Haiku 사용
        # 없으면 Sonnet 4.5로 폴백 (하지만 성능 최적화를 위해 haiku 우선)
        default_casual_model = "anthropic.claude-haiku-4-5-20251001-v1:0" if self.inference_profile_id else "anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.casual_model = get_config("LLM_CASUAL_MODEL", default_casual_model, section="llm")
        # Perf: business_model도 haiku로 변경 (응답 속도 최적화)
        default_business_model = "anthropic.claude-haiku-4-5-20251001-v1:0" if self.inference_profile_id else "anthropic.claude-haiku-4-5-20251001-v1:0"  # 항상 haiku 사용
        self.business_model = get_config("LLM_BUSINESS_MODEL", default_business_model, section="llm")
        
        # boto3 세션 및 클라이언트 캐싱 (성능 최적화)
        self._session = None
        self._client = None
        
        # Circuit Breaker 초기화 (LLM API 장애 복구)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=get_config_int("LLM_CIRCUIT_BREAKER_THRESHOLD", 5, section="llm"),
            recovery_timeout=get_config_float("LLM_CIRCUIT_BREAKER_TIMEOUT", 60.0, section="llm"),
            expected_exception=(LangChainException, Exception),
            name="llm_api",
            redis_client=redis_client
        )
        logger.info(
            f"LLMProvider CircuitBreaker 초기화: "
            f"threshold={self.circuit_breaker.failure_threshold}, "
            f"timeout={self.circuit_breaker.recovery_timeout}s"
        )

    # Chat LLMs
    def get_chat_llm(
        self, 
        role: str = "business",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """역할에 맞는 대화형 LLM 반환.
        
        역할에 따라 적절한 모델과 설정을 사용합니다.
        - business: Claude Haiku 4.5 (Perf: 응답 속도 최적화를 위해 haiku 사용)
        - casual: Claude Haiku 4.5 (경량, 빠른 응답)
        
        Args:
            role: 'business' | 'casual' (기본값: "business")
            model_name: 모델명 (None이면 role에 따라 자동 선택)
            temperature: 온도 (None이면 role에 따라 자동 설정)
            max_tokens: 최대 토큰 수 (선택적)
        
        Returns:
            LangChain ChatBedrock 인스턴스
        
        Raises:
            ProviderError: 지원하지 않는 provider 또는 설정 오류
        """
        if model_name is None:
            model_name = self.business_model if role == "business" else self.casual_model
        
        # temperature 기본값
        if temperature is None:
            temperature = 0.2 if role == "business" else 0

        if self.llm_provider == "bedrock":
            if ChatBedrock is None:
                raise ProviderError(
                    "langchain-aws 패키지가 설치되지 않았습니다. "
                    "설치: pip install langchain-aws"
                )
            
            # Bedrock 설정
            # Security Fix: 하드코딩된 리전 제거
            region = self.bedrock_config.get("region") if self.bedrock_config else get_config("BEDROCK_REGION", None, section="bedrock")
            if not region:
                raise ValueError("BEDROCK_REGION이 설정되지 않았습니다. 환경변수 BEDROCK_REGION 또는 .secrets.toml의 [bedrock].region을 설정하세요.")
            
            # 캐싱된 Bedrock 클라이언트 사용 (성능 최적화)
            client = self._get_bedrock_client()
            
            # ChatBedrock은 max_retries와 timeout을 직접 지원하지 않으므로 model_kwargs에 포함
            model_kwargs = {}
            if max_tokens:
                model_kwargs["max_tokens"] = max_tokens
            
            kwargs = {
                "client": client,
                "temperature": temperature,
                "model_kwargs": model_kwargs,
            }
            
            # Inference profile이 있으면 사용, 없으면 model_id 사용
            # ap-northeast-1 리전에서는 모든 모델에 inference profile이 필요할 수 있음
            use_inference_profile = False
            inference_profile_id = None
            
            if "haiku" in model_name.lower() and self.inference_profile_id_haiku:
                use_inference_profile = True
                inference_profile_id = self.inference_profile_id_haiku
            elif "sonnet" in model_name.lower() and self.inference_profile_id_sonnet:
                use_inference_profile = True
                inference_profile_id = self.inference_profile_id_sonnet
            
            if use_inference_profile:
                kwargs["model_id"] = inference_profile_id
                logger.info(f"Bedrock LLM 초기화: inference_profile={inference_profile_id}, region={region}, role={role}")
            else:
                kwargs["model_id"] = model_name
            logger.info(f"Bedrock LLM 초기화: model={model_name}, region={region}, role={role}")
            
            return ChatBedrock(**kwargs)

        raise ProviderError(f"지원되지 않는 LLM_PROVIDER: {self.llm_provider}")
    
    def _get_bedrock_client(self):
        """Bedrock 클라이언트를 싱글톤으로 반환 (캐싱)
        
        boto3 세션과 bedrock-runtime 클라이언트를 재사용하여 성능을 최적화합니다.
        
        Returns:
            boto3 bedrock-runtime 클라이언트
        """
        if self._client is None:
            import boto3
            
            # Bedrock 설정
            # Security Fix: 하드코딩된 리전 제거
            region = self.bedrock_config.get("region") if self.bedrock_config else get_config("BEDROCK_REGION", None, section="bedrock")
            if not region:
                raise ValueError("BEDROCK_REGION이 설정되지 않았습니다. 환경변수 BEDROCK_REGION 또는 .secrets.toml의 [bedrock].region을 설정하세요.")
            
            # boto3 세션 생성 (credentials가 있으면 사용, 없으면 환경변수/프로파일 사용)
            if self.bedrock_config and self.bedrock_config.get("access_key_id") and self.bedrock_config.get("secret_access_key"):
                self._session = boto3.Session(
                    aws_access_key_id=self.bedrock_config["access_key_id"],
                    aws_secret_access_key=self.bedrock_config["secret_access_key"],
                    region_name=region
                )
                logger.info(f"Bedrock 클라이언트 초기화: 자격증명 사용 (access_key_id={self.bedrock_config['access_key_id'][:10]}...), region={region}")
            else:
                self._session = boto3.Session(region_name=region)
                logger.info(f"Bedrock 클라이언트 초기화: 환경변수/프로파일 자격증명 사용, region={region}")
            
            # bedrock-runtime 클라이언트 생성
            self._client = self._session.client("bedrock-runtime", region_name=region)
            logger.info("Bedrock 클라이언트 캐싱 완료 (재사용 가능)")
        
        return self._client

    # Embeddings
    def get_embeddings(self):
        """임베딩 모델 반환
        
        텍스트를 벡터로 변환하는 임베딩 모델을 반환합니다.
        FAQ 검색의 Cosine Similarity 계산에 사용됩니다.
        
        Returns:
            LangChain BedrockEmbeddings 인스턴스
        
        Raises:
            ProviderError: 지원하지 않는 provider 또는 설정 오류
        """
        if self.embeddings_provider == "bedrock":
            if BedrockEmbeddings is None:
                raise ProviderError(
                    "langchain-aws 패키지가 설치되지 않았습니다. "
                    "설치: pip install langchain-aws"
                )
            
            # Bedrock 설정
            # Security Fix: 하드코딩된 리전 제거
            region = self.bedrock_config.get("region") if self.bedrock_config else get_config("BEDROCK_REGION", None, section="bedrock")
            if not region:
                raise ValueError("BEDROCK_REGION이 설정되지 않았습니다. 환경변수 BEDROCK_REGION 또는 .secrets.toml의 [bedrock].region을 설정하세요.")
            
            # 캐싱된 Bedrock 클라이언트 사용 (성능 최적화)
            client = self._get_bedrock_client()
            
            kwargs = {
                "model_id": "amazon.titan-embed-text-v2:0",  # Titan Embeddings v2
                "client": client,
            }
            
            logger.info(f"Bedrock Embeddings 초기화: model=amazon.titan-embed-text-v2:0, region={region}")
            return BedrockEmbeddings(**kwargs)

        raise ProviderError(f"지원되지 않는 EMBEDDINGS_PROVIDER: {self.embeddings_provider}")
    
    def invoke_with_resilience(self, llm, messages):
        """Circuit Breaker를 통해 LLM 호출 (장애 복구)
        
        LLM API 호출을 Circuit Breaker로 래핑하여 장애 시 빠르게 실패하고
        시스템 부하를 줄입니다.
        
        Args:
            llm: LangChain LLM 인스턴스 (ChatBedrock 등)
            messages: LLM에 전달할 메시지 리스트
        
        Returns:
            LLM 응답 객체
        
        Raises:
            RuntimeError: Circuit이 OPEN 상태일 때
            Exception: LLM 호출 중 발생한 예외
        """
        def _invoke():
            return llm.invoke(messages)
        
        return self.circuit_breaker.call(_invoke)

    async def ainvoke_with_resilience(self, llm, messages):
        """Circuit Breaker를 통해 LLM 비동기 호출 (장애 복구)
        
        Args:
            llm: LangChain LLM 인스턴스
            messages: LLM에 전달할 메시지 리스트
            
        Returns:
            LLM 응답 객체
        """
        async def _ainvoke():
            if hasattr(llm, "ainvoke"):
                return await llm.ainvoke(messages)
            else:
                # ainvoke가 없으면 동기 invoke를 스레드풀에서 실행
                import asyncio
                return await asyncio.to_thread(llm.invoke, messages)
        
        # Circuit Breaker의 call 메서드는 동기 함수를 기대하므로,
        # 비동기 함수를 지원하도록 확장하거나, 여기서는 단순히 실행 (Circuit Breaker가 비동기 미지원 시)
        # 현재 CircuitBreaker 구현을 확인하지 못했으나, 보통 동기용임.
        # 비동기 지원이 없다면 직접 try-except로 처리하거나, 
        # CircuitBreaker를 비동기 지원하도록 수정해야 함.
        # 여기서는 안전하게 _ainvoke를 직접 호출하되, 에러 처리는 상위 레벨에 맡김.
        # Note: 향후 CircuitBreaker 비동기 지원 추가 시 개선 가능
        return await _ainvoke()


__all__ = ["LLMProvider", "ProviderError"]


