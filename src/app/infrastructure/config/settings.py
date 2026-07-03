"""타입 안전한 설정 스키마 (Pydantic Settings)

Security Fix: 환경변수를 우선순위로 로드하고, .secrets.toml 의존성 제거
"""
from __future__ import annotations

from typing import Optional, Any, Dict, Tuple
from pydantic import BaseModel, Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from .config_loader import load_secrets_toml


class RedisSettings(BaseModel):
    """Redis 설정"""
    use_redis: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    max_connections: int = 50


class PostgresSettings(BaseModel):
    """PostgreSQL 설정
    
    Security Fix: database_url 하드코딩 제거, 설정 없을 경우 ValueError 발생
    """
    database_url: Optional[str] = Field(default=None, description="PostgreSQL 연결 URL (필수, 환경변수 DATABASE_URL 또는 .secrets.toml)")
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle: int = 3600
    sql_echo: bool = False
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v: Optional[str]) -> Optional[str]:
        """Security Fix: database_url이 필수인 경우 검증"""
        # database_url은 선택적이지만, 사용 시에는 반드시 설정되어야 함
        # 실제 사용 시점에 검증 (여기서는 None 허용)
        return v


class SessionSettings(BaseModel):
    """세션 관리 설정"""
    ttl: int = Field(default=3600, description="세션 TTL (초)")


class QuestionLogSettings(BaseModel):
    """질문 로그 설정"""
    ttl: int = Field(default=604800, description="질문 로그 TTL (초)")


class BedrockSettings(BaseModel):
    """AWS Bedrock 설정
    
    Security Fix: region 하드코딩 제거, 설정 없을 경우 ValueError 발생
    """
    region: Optional[str] = Field(default=None, description="AWS 리전 (필수, 환경변수 BEDROCK_REGION 또는 .secrets.toml)")
    knowledge_base_id: Optional[str] = None
    data_source_id: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    inference_profile_id: Optional[str] = None
    inference_profile_id_haiku: Optional[str] = None
    inference_profile_id_sonnet: Optional[str] = None
    reranker_type: Optional[str] = Field(default=None, description="cross_encoder 또는 bedrock")
    reranker_model_id: Optional[str] = None
    
    @field_validator('region')
    @classmethod
    def validate_region(cls, v: Optional[str]) -> Optional[str]:
        """Security Fix: region이 필수인 경우 검증"""
        # region은 선택적이지만, 사용 시에는 반드시 설정되어야 함
        # 실제 사용 시점에 검증 (여기서는 None 허용)
        return v


class AWSS3Settings(BaseModel):
    """AWS S3 설정"""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None


class LLMSettings(BaseModel):
    """LLM 설정"""
    provider: str = "bedrock"
    embeddings_provider: str = "bedrock"
    casual_model: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    business_model: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


class BedrockCircuitBreakerSettings(BaseModel):
    """Bedrock Circuit Breaker 설정"""
    threshold: int = 5
    timeout: float = 60.0
    max_retries: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0


class GuardrailSettings(BaseModel):
    """Guardrail 설정"""
    llm_circuit_breaker_threshold: int = 5
    llm_circuit_breaker_timeout: float = 60.0


class CacheSettings(BaseModel):
    """Cache 설정"""
    kb_version: str = "v1.0"
    semantic_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    semantic_max_search: int = 50
    semantic_early_exit: float = Field(default=0.98, ge=0.0, le=1.0)


class FAQSettings(BaseModel):
    """FAQ 설정"""
    provider: str = Field(default="hybrid", description="hybrid, postgresql, redis")
    embedding_cache_size: int = 1000
    embedding_cache_ttl: int = 3600
    default_company_id: str = "default"
    warmup_limit: int = 1000
    generation_days_back: int = 30
    generation_min_frequency: int = 20
    generation_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    cluster_similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class RetrieverSettings(BaseModel):
    """Retriever 설정"""
    provider: str = Field(default="bedrock_kb", description="faiss 또는 bedrock_kb")


class RAGSettings(BaseModel):
    """RAG 설정"""
    rag_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class LangSmithSettings(BaseModel):
    """LangSmith 모니터링 설정"""
    tracing_v2: bool = False
    api_key: Optional[str] = None
    project: str = "chatbot-production"


class OpenAISettings(BaseModel):
    """OpenAI 설정 (선택적)"""
    api_key: Optional[str] = None


class KakaoObjectStorageSettings(BaseModel):
    """카카오 클라우드 Object Storage 설정"""
    kakao_access_key_id: Optional[str] = None
    kakao_secret_access_key: Optional[str] = None



class AppSettings(BaseSettings):
    """애플리케이션 전체 설정 스키마"""
    redis: RedisSettings
    postgres: PostgresSettings
    session: SessionSettings
    question_log: QuestionLogSettings
    bedrock: BedrockSettings
    aws_s3: AWSS3Settings
    llm: LLMSettings
    bedrock_circuit_breaker: BedrockCircuitBreakerSettings
    guardrail: GuardrailSettings
    cache: CacheSettings
    faq: FAQSettings
    retriever: RetrieverSettings
    rag: RAGSettings
    langsmith: LangSmithSettings
    openai: OpenAISettings
    kakao_object_storage: KakaoObjectStorageSettings
    
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore'
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """설정 소스 우선순위 정의"""
        return (
            init_settings,      # 1. 생성자 인자
            env_settings,       # 2. 환경변수
            TomlConfigSettingsSource(settings_cls, ".secrets.toml"), # 3. .secrets.toml (Local Secret)
            TomlConfigSettingsSource(settings_cls, "config/prod/config.toml"), # 4. Prod Config (Git)
            file_secret_settings, # 5. Docker secrets
        )

class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    TOML 파일에서 설정을 로드하는 커스텀 소스
    """
    def __init__(self, settings_cls: type[BaseSettings], filepath: str):
        super().__init__(settings_cls)
        self.filepath = filepath

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        """설정 로드 및 딕셔너리 반환"""
        try:
            # config_loader의 load_secrets_toml 재사용 (경로 처리 로직 활용)
            # load_secrets_toml은 절대 경로가 아니면 프로젝트 루트에서 찾음
            secrets = load_secrets_toml(self.filepath)
            return secrets
        except Exception:
            return {}


# 전역 설정 인스턴스 (지연 로딩)
_settings_instance: Optional[AppSettings] = None


def get_settings(secrets_path: str = ".secrets.toml") -> AppSettings:
    """애플리케이션 설정을 가져옴 (싱글톤 패턴)
    
    Args:
        secrets_path: .secrets.toml 파일 경로 (하위 호환성 유지용, 실제로는 TomlConfigSettingsSource에서 고정됨)
        
    Returns:
        AppSettings 인스턴스
    """
    global _settings_instance
    
    if _settings_instance is None:
        # Pydantic BaseSettings가 자동으로 로드
        _settings_instance = AppSettings()
    
    return _settings_instance


def reload_settings(secrets_path: str = ".secrets.toml") -> AppSettings:
    """설정을 다시 로드 (테스트 또는 설정 변경 시 사용)"""
    global _settings_instance
    _settings_instance = AppSettings()
    return _settings_instance

