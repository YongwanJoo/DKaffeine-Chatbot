"""데이터베이스 세션 관리"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
import logging

from app.infrastructure.persistence.models.base import Base
from app.infrastructure.persistence.models.chatbot_config_model import ChatbotConfig  # noqa: F401
from app.infrastructure.persistence.models.faq_model import Faq  # noqa: F401
from app.infrastructure.config.config_loader import get_config, get_config_int, get_config_bool

logger = logging.getLogger(__name__)

# 데이터베이스 URL (psycopg3 사용)
# Security Fix: 하드코딩된 비밀번호 제거, 환경변수 우선순위, 설정 없을 경우 ValueError
# 우선순위: 환경변수 DATABASE_URL > .secrets.toml > ValueError (설정 필수)
DATABASE_URL = get_config(
    "database_url",  # 소문자로 통일 (.secrets.toml 키와 일치)
    None,  # Security Fix: 기본값 제거 (설정 필수)
    section="postgres"
)

# Security Fix: DATABASE_URL이 없으면 에러 발생
if not DATABASE_URL:
    import sys
    logger.error("DATABASE_URL이 설정되지 않았습니다. 환경변수 DATABASE_URL 또는 .secrets.toml의 [postgres].database_url을 설정하세요.")
    sys.exit(1)

# Connection Pool 설정
POOL_SIZE = get_config_int("DB_POOL_SIZE", 10, section="postgres")
MAX_OVERFLOW = get_config_int("DB_MAX_OVERFLOW", 20, section="postgres")
POOL_RECYCLE = get_config_int("DB_POOL_RECYCLE", 3600, section="postgres")  # 1시간

# Engine 생성 (Connection Pool 사용)
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,  # 연결 유효성 검사
    pool_recycle=POOL_RECYCLE,  # 1시간마다 연결 재생성
    echo=get_config_bool("SQL_ECHO", False, section="postgres")
)

# Fix: DB 연결 실패 시 sys.exit(1)로 프로세스 종료
try:
    with engine.connect() as conn:
        # Fix: SQLAlchemy 2.0 스타일 - text() 함수로 래핑 필요
        conn.execute(text("SELECT 1"))
    logger.info(
        f"Database engine initialized: pool_size={POOL_SIZE}, "
        f"max_overflow={MAX_OVERFLOW}, pool_recycle={POOL_RECYCLE}s"
    )
except Exception as e:
    import sys
    logger.error(f"Database connection failed: {e}")
    logger.error("애플리케이션을 종료합니다.")
    sys.exit(1)

# Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """데이터베이스 테이블 생성
    
    Fix: DB 마이그레이션은 외부 스크립트로 분리 (Alembic 등)
    이 함수는 제거되었으며, DB 마이그레이션은 별도 프로세스로 실행해야 합니다.
    
    배포 환경: 모든 테이블과 외래키 제약조건 생성
    테스트 환경: 외래키 제약조건이 실패해도 계속 진행 (테이블이 없을 수 있음)
    """
    from sqlalchemy.exc import IntegrityError, ProgrammingError
    import psycopg.errors as pg_errors
    
    try:
        Base.metadata.create_all(bind=engine)
    except (IntegrityError, ProgrammingError) as e:
        # Fix: 문자열 기반 에러 처리 제거, SQLAlchemy 예외 클래스 사용
        # PostgreSQL 에러 코드 확인
        if hasattr(e.orig, 'pgcode'):
            pgcode = e.orig.pgcode
            # 외래키 제약조건 관련 에러 (23503: foreign_key_violation)
            if pgcode == '23503' or pgcode == '42P01':  # foreign_key_violation or undefined_table
                logger.warning(
                    f"외래키 제약조건 생성 실패 (테스트 환경일 수 있음): {e}. "
                    f"테이블은 생성되었지만 외래키 제약조건은 건너뜁니다."
                )
            else:
                # 다른 에러는 다시 발생시킴
                raise
        else:
            # pgcode가 없는 경우 문자열 기반 폴백 (하위 호환성)
            error_msg = str(e).lower()
            if "foreign key" in error_msg or "noreferencedtable" in error_msg or "category" in error_msg or "chat_model" in error_msg:
                logger.warning(
                    f"외래키 제약조건 생성 실패 (테스트 환경일 수 있음): {e}. "
                    f"테이블은 생성되었지만 외래키 제약조건은 건너뜁니다."
                )
            else:
                raise


def get_db() -> Generator[Session, None, None]:
    """데이터베이스 세션 생성 (의존성 주입용)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """컨텍스트 매니저로 데이터베이스 세션 사용"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# 비동기 데이터베이스 세션 (AsyncSession용)
# psycopg를 async 모드로 사용하려면 postgresql+psycopg_async:// 사용
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql+psycopg_async://")

# Async Engine 생성
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=POOL_RECYCLE,
    echo=get_config_bool("SQL_ECHO", False, section="postgres")
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

logger.info("Async database engine initialized")


@asynccontextmanager
async def async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """비동기 컨텍스트 매니저로 데이터베이스 세션 사용"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

