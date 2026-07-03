"""설정 파일 로더 (config.toml + .secrets.toml + 환경변수 지원)

배포 환경 지원: 환경변수 우선, 없으면 .secrets.toml > config.toml > 기본값
환경변수는 배포 환경(Docker, Kubernetes 등)에서 설정 관리에 유용

설정 파일 우선순위:
1. 환경변수 (최우선)
2. .secrets.toml (민감 정보)
3. config.toml (일반 설정)
4. 기본값

하위 호환성: config.toml이 없어도 .secrets.toml만으로 동작
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        tomllib = None

logger = logging.getLogger(__name__)

# 전역 설정 캐시 (한 번만 로드)
_config_cache: Optional[Dict[str, Any]] = None

# 프로젝트 루트 디렉토리 경로 (config_loader.py 기준: src/app/infrastructure/config/config_loader.py)
# 프로젝트 루트는 config_loader.py에서 5단계 위 (src/app/infrastructure/config -> 프로젝트 루트)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


def load_config_toml(config_path: str | Path = "config.toml") -> Dict[str, Any]:
    """`config.toml` 파일을 읽어서 딕셔너리로 반환 (일반 설정)
    
    Args:
        config_path: config.toml 파일 경로
        
    Returns:
        TOML 파일 내용을 딕셔너리로 변환한 결과 (파일이 없으면 빈 딕셔너리)
        
    Raises:
        ValueError: tomllib를 사용할 수 없을 때
    """
    if tomllib is None:
        raise ValueError(
            "tomllib를 사용할 수 없습니다. Python 3.11+ 또는 tomli 패키지 설치가 필요합니다.\n"
            "설치: pip install tomli"
        )
    
    path = Path(config_path)
    
    # 절대 경로가 아니면 프로젝트 루트 기준으로 찾기
    if not path.is_absolute():
        # 1. 현재 작업 디렉토리에서 찾기
        if not path.exists():
            # 2. 프로젝트 루트에서 찾기
            project_path = _PROJECT_ROOT / path.name
            if project_path.exists():
                path = project_path
            else:
                logger.debug(f"config.toml 파일이 없습니다: {config_path} (하위 호환성: .secrets.toml만 사용)")
                return {}
    
    if not path.exists():
        logger.debug(f"config.toml 파일이 없습니다: {path} (하위 호환성: .secrets.toml만 사용)")
        return {}
    
    try:
        with path.open("rb") as f:
            config_data = tomllib.load(f)
            logger.info(f"config.toml 로드 완료: {path}")
            return config_data
    except Exception as e:
        logger.warning(f"config.toml 로드 실패: {path}, 에러: {e}")
        return {}


def load_secrets_toml(secrets_path: str | Path = ".secrets.toml") -> Dict[str, Any]:
    """`.secrets.toml` 파일을 읽어서 딕셔너리로 반환 (민감 정보)
    
    Args:
        secrets_path: .secrets.toml 파일 경로
    
    Returns:
        TOML 파일 내용을 딕셔너리로 변환한 결과 (파일이 없으면 빈 딕셔너리)
        
    Raises:
        ValueError: tomllib를 사용할 수 없을 때
    """
    if tomllib is None:
        raise ValueError(
            "tomllib를 사용할 수 없습니다. Python 3.11+ 또는 tomli 패키지 설치가 필요합니다.\n"
            "설치: pip install tomli"
        )
    
    path = Path(secrets_path)
    
    # 절대 경로가 아니면 프로젝트 루트 기준으로 찾기
    if not path.is_absolute():
        # 1. 현재 작업 디렉토리에서 찾기
        if not path.exists():
            # 2. 프로젝트 루트에서 찾기
            project_path = _PROJECT_ROOT / path.name
            if project_path.exists():
                path = project_path
            else:
                logger.warning(
                    f".secrets.toml 파일을 찾을 수 없습니다. "
                    f"시도한 경로: {Path(secrets_path).absolute()}, {project_path}"
                )
                return {}
    
    if not path.exists():
        logger.warning(
            f".secrets.toml 파일이 없습니다: {path}. "
            f"프로젝트 루트({_PROJECT_ROOT})에 파일이 있는지 확인하세요."
        )
        return {}
    
    try:
        with path.open("rb") as f:
            secrets_data = tomllib.load(f)
            logger.info(f".secrets.toml 로드 완료: {path} ({len(secrets_data)} 섹션)")
            # 로드된 섹션 목록 로깅 (디버깅용)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"로드된 섹션: {list(secrets_data.keys())}")
            return secrets_data
    except Exception as e:
        logger.error(f".secrets.toml 로드 실패: {path}, 에러: {e}", exc_info=True)
        return {}


def _merge_configs(config_data: Dict[str, Any], secrets_data: Dict[str, Any]) -> Dict[str, Any]:
    """config.toml과 .secrets.toml을 병합
    
    우선순위: .secrets.toml > config.toml (같은 키가 있으면 .secrets.toml이 우선)
    섹션별로도 병합 (예: [postgres] 섹션)
    
    Args:
        config_data: config.toml 데이터
        secrets_data: .secrets.toml 데이터
        
    Returns:
        병합된 설정 딕셔너리
    """
    merged = config_data.copy()
    
    # .secrets.toml의 모든 섹션을 병합 (덮어쓰기)
    for section_name, section_data in secrets_data.items():
        if isinstance(section_data, dict):
            if section_name in merged and isinstance(merged[section_name], dict):
                # 섹션이 이미 있으면 병합
                merged[section_name] = {**merged[section_name], **section_data}
            else:
                # 섹션이 없으면 추가
                merged[section_name] = section_data.copy()
        else:
            # 섹션이 dict가 아니면 그대로 덮어쓰기
            merged[section_name] = section_data
    
    return merged


def get_config(key: str, default: Any = None, section: Optional[str] = None) -> Any:
    """설정 값을 환경변수 또는 설정 파일에서 읽어옴
    
    우선순위: 환경변수 > .secrets.toml > config.toml > 기본값
    배포 환경(Docker, Kubernetes 등)에서 환경변수 사용 권장
    
    Args:
        key: 설정 키 (설정 파일의 키 또는 환경변수 이름)
        default: 기본값 (키가 없을 때 반환)
        section: 설정 파일의 섹션 이름 (예: "redis", "bedrock")
    
    Returns:
        설정 값
    
    Raises:
        ValueError: 설정 파일을 읽을 수 없을 때 (tomllib 없음)
    """
    global _config_cache
    
    # 1. 환경변수 확인 (배포 환경 우선)
    env_key = _get_env_key(key, section)
    if env_key and env_key in os.environ:
        value = os.environ[env_key]
        logger.debug(f"환경변수에서 설정 로드: {env_key}={value[:20] if isinstance(value, str) and len(value) > 20 else value}")
        return value
    
    # 2. 설정 파일에서 읽기 (config.toml + .secrets.toml 병합)
    if _config_cache is None:
        try:
            # config.toml 로드 (일반 설정)
            config_data = load_config_toml()
            # .secrets.toml 로드 (민감 정보)
            secrets_data = load_secrets_toml()
            # 병합: .secrets.toml이 우선 (덮어쓰기)
            _config_cache = _merge_configs(config_data, secrets_data)
            logger.info(
                f"설정 파일 로드 완료: config.toml ({len(config_data)} 섹션), "
                f".secrets.toml ({len(secrets_data)} 섹션), "
                f"병합된 설정 ({len(_config_cache)} 섹션)"
            )
            # Bedrock 설정 확인 (디버깅용)
            if "bedrock" in _config_cache:
                bedrock_config = _config_cache["bedrock"]
                logger.debug(
                    f"Bedrock 설정 확인: "
                    f"region={bedrock_config.get('region')}, "
                    f"knowledge_base_id={'설정됨' if bedrock_config.get('knowledge_base_id') else '없음'}, "
                    f"reranker_type={bedrock_config.get('reranker_type')}, "
                    f"reranker_model_id={bedrock_config.get('reranker_model_id')}"
                )
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}", exc_info=True)
            _config_cache = {}
    
    if section:
        # 특정 섹션에서 찾기
        section_data = _config_cache.get(section, {})
        if isinstance(section_data, dict) and key in section_data:
            return section_data[key]
    
    # 섹션 없이 전체에서 찾기 (하위 호환성)
    for sec_name, sec_data in _config_cache.items():
        if isinstance(sec_data, dict) and key in sec_data:
            return sec_data[key]
    
    # 3. 기본값 반환
    return default


def _get_env_key(key: str, section: Optional[str] = None) -> Optional[str]:
    """환경변수 키 이름 생성
    
    섹션과 키를 조합하여 환경변수 이름 생성
    예: section="redis", key="host" -> "REDIS_HOST"
    예: section="postgres", key="database_url" -> "POSTGRES_DATABASE_URL" 또는 "DATABASE_URL"
    
    Args:
        key: 설정 키 (소문자 또는 이미 대문자로 된 키 모두 지원)
        section: 섹션 이름
    
    Returns:
        환경변수 이름 (대문자, 언더스코어 구분)
    """
    # 특수 케이스: database_url은 DATABASE_URL로 매핑 (일반적인 관례)
    if key == "database_url" and section == "postgres":
        return "DATABASE_URL"
    
    # 이미 대문자로 된 키인지 확인 (하위 호환성)
    # 예: "BEDROCK_CIRCUIT_BREAKER_THRESHOLD" -> 그대로 사용
    if key.isupper() and "_" in key:
        # 이미 환경변수 형식인 경우 그대로 반환 (섹션 무시)
        return key
    
    # 섹션이 있으면 섹션_키 형식
    if section:
        return f"{section.upper()}_{key.upper()}"
    
    # 섹션이 없으면 키만 사용
    return key.upper()


def get_config_bool(key: str, default: bool = False, section: Optional[str] = None) -> bool:
    """불린 설정 값 읽기"""
    value = get_config(key, default, section)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def get_config_int(key: str, default: int = 0, section: Optional[str] = None) -> int:
    """정수 설정 값 읽기"""
    value = get_config(key, default, section)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def get_config_float(key: str, default: float = 0.0, section: Optional[str] = None) -> float:
    """실수 설정 값 읽기"""
    value = get_config(key, default, section)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def get_all_config() -> Dict[str, Any]:
    """모든 설정을 딕셔너리로 반환 (.secrets.toml 전용)
    
    Returns:
        모든 설정을 담은 딕셔너리
    """
    config = {}
    
    # Redis 설정
    config["redis"] = {
        "use_redis": get_config_bool("use_redis", False, "redis"),
        "host": get_config("host", "localhost", "redis"),
        "port": get_config_int("port", 6379, "redis"),
        "db": get_config_int("db", 0, "redis"),
        "max_connections": get_config_int("max_connections", 50, "redis"),
    }
    
    # PostgreSQL 설정
    config["postgres"] = {
        "database_url": get_config("database_url", None, "postgres"),
        "pool_size": get_config_int("pool_size", 10, "postgres"),
        "max_overflow": get_config_int("max_overflow", 20, "postgres"),
        "pool_recycle": get_config_int("pool_recycle", 3600, "postgres"),
        "sql_echo": get_config_bool("sql_echo", False, "postgres"),
    }
    
    # 세션 설정
    config["session"] = {
        "ttl": get_config_int("ttl", 3600, "session"),
    }
    
    # 질문 로그 설정
    config["question_log"] = {
        "ttl": get_config_int("ttl", 604800, "question_log"),
    }
    
    # Bedrock 설정
    config["bedrock"] = get_bedrock_config()
    
    # LLM 설정
    config["llm"] = {
        "provider": get_config("provider", "bedrock", "llm"),
        "embeddings_provider": get_config("embeddings_provider", "bedrock", "llm"),
        "casual_model": get_config("casual_model", "anthropic.claude-haiku-4-5-20251001-v1:0", "llm"),
        "business_model": get_config("business_model", "anthropic.claude-sonnet-4-5-20250929-v1:0", "llm"),
        "circuit_breaker_threshold": get_config_int("circuit_breaker_threshold", 5, "llm"),
        "circuit_breaker_timeout": get_config_float("circuit_breaker_timeout", 60.0, "llm"),
    }
    
    # Bedrock Circuit Breaker 설정
    config["bedrock_circuit_breaker"] = {
        "threshold": get_config_int("threshold", 5, "bedrock_circuit_breaker"),
        "timeout": get_config_float("timeout", 60.0, "bedrock_circuit_breaker"),
        "max_retries": get_config_int("max_retries", 3, "bedrock_circuit_breaker"),
        "retry_initial_delay": get_config_float("retry_initial_delay", 1.0, "bedrock_circuit_breaker"),
        "retry_max_delay": get_config_float("retry_max_delay", 60.0, "bedrock_circuit_breaker"),
    }
    
    # Guardrail 설정
    config["guardrail"] = {
        "llm_circuit_breaker_threshold": get_config_int("llm_circuit_breaker_threshold", 5, "guardrail"),
        "llm_circuit_breaker_timeout": get_config_float("llm_circuit_breaker_timeout", 60.0, "guardrail"),
    }
    
    # Cache 설정
    config["cache"] = {
        "kb_version": get_config("kb_version", "v1.0", "cache"),
        "semantic_similarity_threshold": get_config_float("semantic_similarity_threshold", 0.95, "cache"),
        "semantic_max_search": get_config_int("semantic_max_search", 50, "cache"),
        "semantic_early_exit": get_config_float("semantic_early_exit", 0.98, "cache"),
    }
    
    # FAQ 설정
    config["faq"] = {
        "provider": get_config("provider", "hybrid", "faq"),
        "embedding_cache_size": get_config_int("embedding_cache_size", 1000, "faq"),
        "embedding_cache_ttl": get_config_int("embedding_cache_ttl", 3600, "faq"),
        "default_company_id": get_config("default_company_id", "default", "faq"),
        "warmup_limit": get_config_int("warmup_limit", 1000, "faq"),
        "generation_days_back": get_config_int("generation_days_back", 30, "faq"),
        "generation_min_frequency": get_config_int("generation_min_frequency", 20, "faq"),
        "generation_min_confidence": get_config_float("generation_min_confidence", 0.85, "faq"),
        "cluster_similarity_threshold": get_config_float("cluster_similarity_threshold", 0.8, "faq"),
    }
    
    # Retriever 설정
    config["retriever"] = {
        "provider": get_config("provider", "bedrock_kb", "retriever"),
    }
    
    # LangSmith 설정
    config["langsmith"] = {
        "tracing_v2": get_config_bool("tracing_v2", False, "langsmith"),
        "api_key": get_config("api_key", None, "langsmith"),
        "project": get_config("project", "chatbot-production", "langsmith"),
    }
    
    return config


def get_bedrock_config() -> Dict[str, Optional[str]]:
    """Bedrock KB 설정을 .secrets.toml에서 읽어옴
    
    단일 설정 시스템: .secrets.toml만 사용
    
    Returns:
        Bedrock 설정 딕셔너리
    """
    return {
        "region": get_config("region", None, "bedrock"),
        "knowledge_base_id": get_config("knowledge_base_id", None, "bedrock"),
        "data_source_id": get_config("data_source_id", None, "bedrock"),
        "access_key_id": get_config("access_key_id", None, "bedrock") or get_config("access_key_id", None, "aws_s3"),
        "secret_access_key": get_config("secret_access_key", None, "bedrock") or get_config("secret_access_key", None, "aws_s3"),
        "inference_profile_id": get_config("inference_profile_id", None, "bedrock"),
        "inference_profile_id_haiku": get_config("inference_profile_id_haiku", None, "bedrock"),
        "inference_profile_id_sonnet": get_config("inference_profile_id_sonnet", None, "bedrock"),
        "reranker_type": get_config("reranker_type", None, "bedrock"),
        "reranker_model_id": get_config("reranker_model_id", None, "bedrock"),
    }
