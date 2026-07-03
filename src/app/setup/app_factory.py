"""FastAPI 앱 팩토리"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette_exporter import PrometheusMiddleware, handle_metrics
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from app.infrastructure.config.config_loader import get_config_bool, get_config, get_config_int, get_config_float
from app.domain.exceptions import ChatbotException

from app.presentation.http.controllers import messages_router, health_router

# 프로젝트 루트 디렉토리 경로 (src/app/setup/app_factory.py 기준)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

# 로깅 설정
from app.infrastructure.utils.logging_config import setup_logging, get_logger

# 환경에 따라 로그 레벨 및 형식 결정
LOG_LEVEL = get_config("LOG_LEVEL", "INFO", section="app") or "INFO"
USE_JSON_LOGGING = get_config_bool("USE_JSON_LOGGING", False, section="app")
setup_logging(level=LOG_LEVEL, use_json=USE_JSON_LOGGING)

logger = get_logger(__name__)


def create_services() -> Dict[str, Any]:
    """
    챗봇에 필요한 모든 서비스를 생성하고 반환
    
    Returns:
        Dict[str, Any]: 서비스 인스턴스들을 담은 딕셔너리
    """
    from app.infrastructure.adapters.guardrail import GuardrailService
    from app.infrastructure.adapters.cache import CacheService
    from app.infrastructure.adapters.rag import RAGService
    from app.infrastructure.adapters.llm import LLMProvider
    from app.infrastructure.adapters.llm.llm_adapter import LLMAdapter
    from app.infrastructure.utils.redis_client import get_redis_client
    
    # Redis 클라이언트 가져오기 (CircuitBreaker용)
    redis_client = get_redis_client()
    
    # CacheService용 Redis 사용 여부 설정
    USE_REDIS = get_config_bool("use_redis", False, section="redis")
    
    # LLM Provider 생성 및 어댑터로 래핑 (의존성 주입)
    llm_provider = LLMProvider(redis_client=redis_client)
    llm_adapter = LLMAdapter(llm_provider=llm_provider)
    
    # FAQ 서비스: DB 전용
    from app.infrastructure.adapters.faq import FAQService
    faq_service = FAQService()
    logger.info("FAQ 서비스 초기화 완료 (DB 전용)")
    
    # 뉴스 서비스: 데이터베이스에서 뉴스 조회
    from app.infrastructure.adapters.news import NewsService
    news_service = NewsService()
    logger.info("뉴스 서비스 초기화 완료")
    
    services = {
        "guardrail_service": GuardrailService(llm_provider=llm_adapter, redis_client=redis_client),
        "cache_service": CacheService(use_redis=USE_REDIS),
        "faq_service": faq_service,
        "rag_service": RAGService(redis_client=redis_client),
        "news_service": news_service,
        "provider": llm_provider,  # LLMProvider 인스턴스 재사용
    }
    
    logger.info("모든 서비스 초기화 완료")
    return services


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 앱 생명주기 관리
    
    Startup과 Shutdown을 하나의 컨텍스트 매니저로 관리합니다.
    """
    # ========== Startup ==========
    logger.info("앱 시작: 서비스 초기화 중...")
    
    try:
        # Fix: Heavy Startup 로직 제거
        # - init_db() 제거: DB 마이그레이션은 외부 스크립트로 분리 (Alembic 등)
        # - warmup_faq_cache() 제거: 별도 비동기 프로세스나 배포 후 트리거로 변경
        # - APScheduler 제거: 다중 워커 환경에서 중복 실행 방지, Celery Beat 사용 권장
        
        # 서비스 생성
        services = create_services()
        app.state.services = services
        logger.info("앱 시작: 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}", exc_info=True)
        raise
    
    yield  # 앱 실행 중
    
    # ========== Shutdown ==========
    logger.info("앱 종료: 리소스 정리 시작")
    
    # Fix: 스케줄러 및 워밍업 태스크 제거됨 (lifespan에서 제거)
    
    # RAG 서비스의 reranker 정리
    try:
        if hasattr(app.state, 'services'):
            rag_service = app.state.services.get("rag_service")
            if rag_service and hasattr(rag_service, 'reranker') and rag_service.reranker:
                try:
                    # reranker 모델 정리
                    if hasattr(rag_service.reranker, 'model') and rag_service.reranker.model:
                        rag_service.reranker.model = None
                        logger.info("Reranker 모델 정리 완료")
                except Exception as e:
                    logger.warning(f"Reranker 정리 중 오류: {e}")
    except Exception as e:
        logger.warning(f"서비스 정리 중 오류: {e}")
    
    # 가비지 컬렉션
    try:
        import gc
        gc.collect()
        logger.info("가비지 컬렉션 완료")
    except Exception as e:
        logger.debug(f"가비지 컬렉션 중 오류 (무시 가능): {e}")
    
    # Redis는 TTL로 자동 정리되므로 수동 삭제 불필요
    logger.info("앱 종료: 리소스 정리 완료 (Redis는 TTL로 자동 정리됨)")


def create_app() -> FastAPI:
    """FastAPI 앱 생성 및 설정"""
    app = FastAPI(
        title="B2B RAG 챗봇 API",
        description="LangGraph 기반 챗봇 서비스",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Prometheus 미들웨어 추가 (HTTP 요청 메트릭 자동 수집)
    app.add_middleware(PrometheusMiddleware)
    
    # CORS 설정 (프론트엔드 접근 허용)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 정적 파일 제공 (HTML, CSS, JS) - 선택적 (로컬 개발용)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        logger.info(f"정적 파일 디렉토리: {STATIC_DIR}")
    else:
        logger.debug(f"정적 파일 디렉토리가 없습니다 (선택적): {STATIC_DIR}")
    
    # 요청 검증 오류 핸들러 (422 오류 로깅)
    from app.setup.exception_handlers import validation_exception_handler, chatbot_exception_handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # 커스텀 예외 핸들러 (중앙화된 에러 처리)
    app.add_exception_handler(ChatbotException, chatbot_exception_handler)
    
    from app.presentation.http.controllers import messages_router, health_router, config_router, news_router
    # 라우터 등록
    app.include_router(messages_router)
    app.include_router(health_router)
    app.include_router(config_router)
    app.include_router(news_router)
    
    # Prometheus 메트릭 엔드포인트 추가 (starlette-exporter 사용)
    app.add_route("/metrics", handle_metrics)
    
    # 루트 경로 - 웹 인터페이스 제공 (선택적)
    @app.get("/")
    async def read_root():
        """루트 경로 - 정적 파일이 있으면 제공, 없으면 API 정보 반환"""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        else:
            # 정적 파일이 없으면 API 정보 반환
            return {
                "service": "B2B RAG 챗봇 API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
    
    # Favicon 핸들러 (404 에러 방지)
    @app.get("/favicon.ico")
    async def favicon():
        """Favicon 요청 처리 (404 에러 방지)"""
        from fastapi.responses import Response
        return Response(status_code=204)  # No Content
    
    
    logger.info("FastAPI 앱 생성 완료")
    
    return app

