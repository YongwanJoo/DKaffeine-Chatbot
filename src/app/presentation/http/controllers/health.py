"""헬스체크 컨트롤러

Stability Fix: Liveness Probe와 Readiness Probe 분리
- /health/live: 단순 프로세스 확인 (Liveness)
- /health/ready: DB/Redis 연결 확인 (Readiness)
"""
from fastapi import APIRouter, HTTPException, status
import logging

router = APIRouter(prefix="/api/v1", tags=["health"])

logger = logging.getLogger(__name__)


@router.get("/health/live")
async def liveness_probe():
    """Liveness Probe: 단순 프로세스 확인
    
    Kubernetes Liveness Probe용 엔드포인트
    프로세스가 살아있는지만 확인 (DB/Redis 연결 확인 안 함)
    """
    return {
        "status": "alive",
        "service": "chatbot",
        "version": "1.0.0"
    }


@router.get("/health/ready")
async def readiness_probe():
    """Readiness Probe: DB/Redis 연결 확인
    
    Kubernetes Readiness Probe용 엔드포인트
    DB와 Redis 연결을 확인하여 서비스가 요청을 처리할 준비가 되었는지 확인
    Cascading Failure 방지: DB/Redis 연결 실패 시 503 반환
    """
    checks = {
        "database": False,
        "redis": False
    }
    
    # DB 연결 확인
    try:
        from app.infrastructure.persistence.session import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            # Fix: SQLAlchemy 2.0 스타일 - text() 함수로 래핑 필요
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = False
    
    # Redis 연결 확인
    try:
        from app.infrastructure.utils.redis_client import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            redis_client.ping()
            checks["redis"] = True
        else:
            # Redis가 비활성화된 경우 (use_redis=False)
            checks["redis"] = True  # 비활성화는 정상 상태로 간주
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        checks["redis"] = False
    
    # 모든 체크가 통과했는지 확인
    if not all(checks.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "checks": checks,
                "message": "Service is not ready to handle requests"
            }
        )
    
    return {
        "status": "ready",
        "service": "chatbot",
        "version": "1.0.0",
        "checks": checks
    }


@router.get("/health")
async def health_check():
    """기본 헬스체크 (하위 호환성)
    
    기존 /health 엔드포인트와의 하위 호환성을 위해 유지
    Readiness Probe를 호출하여 DB/Redis 연결 확인
    """
    try:
        result = await readiness_probe()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "message": str(e)
            }
        )

