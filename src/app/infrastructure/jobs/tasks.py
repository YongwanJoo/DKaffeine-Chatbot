"""Celery 백그라운드 작업 정의

채팅 로그 저장 및 FAQ 후보 생성을 위한 Celery Task
비동기 서비스 로직을 동기 Celery Worker에서 실행하기 위해 async_to_sync 사용

Perf: 전역 객체 재사용 (LLMProvider, SimilarityChecker 등)
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from asgiref.sync import async_to_sync

from app.worker import celery_app

logger = logging.getLogger(__name__)

# Perf: 전역 객체 재사용 (워커 초기화 시 한 번만 로드)
_llm_provider = None
_similarity_checker = None
_notification_service = None

def _get_llm_provider():
    """LLMProvider 싱글톤 (워커 초기화 시 한 번만 생성)"""
    global _llm_provider
    if _llm_provider is None:
        from app.infrastructure.adapters.llm import LLMProvider
        from app.infrastructure.utils.redis_client import get_redis_client
        _llm_provider = LLMProvider(redis_client=get_redis_client())
        logger.info("LLMProvider 전역 인스턴스 초기화 완료")
    return _llm_provider

def _get_similarity_checker():
    """SimilarityChecker 싱글톤 (워커 초기화 시 한 번만 생성)"""
    global _similarity_checker
    if _similarity_checker is None:
        from app.infrastructure.adapters.faq.similarity_checker_adapter import SimilarityCheckerAdapter
        _similarity_checker = SimilarityCheckerAdapter()
        logger.info("SimilarityChecker 전역 인스턴스 초기화 완료")
    return _similarity_checker

def _get_notification_service():
    """NotificationService 싱글톤 (워커 초기화 시 한 번만 생성)"""
    global _notification_service
    if _notification_service is None:
        from app.infrastructure.adapters.notification.notification_adapter import NotificationAdapter
        from app.infrastructure.adapters.config.config_adapter import ConfigAdapter
        _notification_service = NotificationAdapter(config=ConfigAdapter())
        logger.info("NotificationService 전역 인스턴스 초기화 완료")
    return _notification_service


@celery_app.task(
    name="save_chat_log",
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 1분 후 재시도
)
def save_chat_log_task(
    self,
    query: str,
    response: str,
    status: str,
    chat_model_id: int,
    input_time_iso: str,  # ISO 8601 형식 문자열
    output_time_iso: str,  # ISO 8601 형식 문자열
    latency_ms: int,
    category_id: Optional[int] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    guardrail_reason: Optional[str] = None
):
    """채팅 로그 저장 Celery Task
    
    Args:
        self: Celery Task 인스턴스 (bind=True)
        query: 사용자 질문
        response: 챗봇 응답
        status: 로그 상태 (SUCCESS, GUARDRAIL, REQUERY, ERROR)
        chat_model_id: 챗봇 모델 ID
        input_time_iso: 입력 시간 (ISO 8601 문자열)
        output_time_iso: 출력 시간 (ISO 8601 문자열)
        latency_ms: 응답 지연 시간 (밀리초)
        category_id: 카테고리 ID (선택)
        prompt_tokens: 프롬프트 토큰 수 (기본값: 0)
        prompt_tokens: 프롬프트 토큰 수 (기본값: 0)
        completion_tokens: 완성 토큰 수 (기본값: 0)
        guardrail_reason: 가드레일 차단 사유 (선택)
    """
    try:
        # ISO 8601 문자열을 datetime 객체로 변환
        input_time = datetime.fromisoformat(input_time_iso)
        output_time = datetime.fromisoformat(output_time_iso)
        
        # 비동기 함수를 동기로 래핑하여 실행
        async_to_sync(_save_chat_log_async)(
            query=query,
            response=response,
            status=status,
            chat_model_id=chat_model_id,
            input_time=input_time,
            output_time=output_time,
            latency_ms=latency_ms,
            category_id=category_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            guardrail_reason=guardrail_reason
        )
        
        logger.info(f"✅ [Celery] 채팅 로그 저장 완료: status={status}, latency={latency_ms}ms, tokens={prompt_tokens}/{completion_tokens}, category_id={category_id}")
        
    except Exception as exc:
        logger.error(f"⚠️ [Celery] 채팅 로그 저장 실패: {exc}", exc_info=True)
        # 재시도 (최대 3회)
        raise self.retry(exc=exc)


@celery_app.task(
    name="generate_faq_candidate",
    bind=True,
    max_retries=2,
    default_retry_delay=120,  # 2분 후 재시도
)
def generate_faq_candidate_task(
    self,
    user_question: str,
    rag_answer: str,
    rag_confidence: float,
    chatbot_settings: Optional[Dict[str, Any]] = None
):
    """FAQ 후보 생성 Celery Task
    
    Polish: 불필요한 async_to_sync와 asyncio.to_thread 제거, 순수 동기 함수로 단순화
    Celery 워커는 이미 별도 프로세스이므로 동기 함수를 직접 실행 가능
    
    Args:
        self: Celery Task 인스턴스 (bind=True)
        user_question: 사용자 질문
        rag_answer: RAG 답변
        rag_confidence: RAG 신뢰도
        chatbot_settings: 챗봇 설정 (선택)
    """
    # Task 실행 시작 로그 (가장 먼저 출력)
    logger.info(
        f"🚀 [Celery] Task 실행 시작: task_id={self.request.id}, "
        f"question='{user_question[:50]}...', confidence={rag_confidence:.3f}"
    )
    
    try:
        from app.infrastructure.persistence.session import db_session
        from app.domain.services.faq_generation_service import FAQGenerationService
        from app.infrastructure.adapters.persistence.faq_repository_adapter import FAQRepositoryAdapter
        
        # Polish: 순수 동기 실행 (불필요한 컨텍스트 스위칭 제거)
        with db_session() as db:
            faq_repo = FAQRepositoryAdapter(db)
            
            # Perf: 전역 객체 재사용 (매번 생성하지 않음)
            llm_provider = _get_llm_provider()
            similarity_checker = _get_similarity_checker()
            
            # Fix: QuestionLogService 주입 (빈도수 체크를 위해 필수)
            from app.infrastructure.adapters.cache.question_log_service import QuestionLogService
            from app.infrastructure.utils.redis_client import get_redis_client
            
            question_log_service = QuestionLogService(redis_client=get_redis_client())
            
            # NotificationService 주입
            notification_service = _get_notification_service()
            
            faq_gen_service = FAQGenerationService(
                faq_repository=faq_repo,
                llm_provider=llm_provider,
                similarity_checker=similarity_checker,
                question_log_service=question_log_service,
                notification_service=notification_service,
                db_session=db
            )
            
            # Polish: 동기 함수를 직접 호출 (훨씬 빠르고 직관적)
            # Settings에서 임계값 추출 (있으면 사용)
            min_confidence = 0.5
            if chatbot_settings and "thresholds" in chatbot_settings:
                min_confidence = chatbot_settings["thresholds"].get("faq_min_confidence", 0.5)

            # Fix: generate_faq_candidate는 async 메서드이므로 async_to_sync 사용
            # Fix: 실시간 생성 시에도 빈도수 조건(10회) 적용
            from app.domain.constants import FAQ_MIN_FREQUENCY
            
            logger.info(f"🔄 [Celery] FAQ 후보 생성 시작: question='{user_question[:50]}...', confidence={rag_confidence}, min_frequency={FAQ_MIN_FREQUENCY}")
            
            result = async_to_sync(faq_gen_service.generate_faq_candidate)(
                user_question=user_question,
                rag_answer=rag_answer,
                rag_confidence=rag_confidence,
                min_confidence=min_confidence,
                min_frequency=FAQ_MIN_FREQUENCY  # 10회 이상일 때만 생성
            )
            
            success, faq_id, message = result
            if success:
                logger.info(f"✅ [Celery] FAQ 후보 생성 완료: confidence={rag_confidence}, faq_id={faq_id}, message={message}")
            else:
                logger.warning(f"⚠️ [Celery] FAQ 후보 생성 실패: {message} (빈도수 부족 또는 기타 이유)")
        
        logger.info(f"✅ [Celery] FAQ 후보 생성 Task 완료: confidence={rag_confidence}")
        
    except Exception as exc:
        logger.error(f"⚠️ [Celery] FAQ 후보 생성 실패: {exc}", exc_info=True)
        # 재시도 (최대 2회)
        raise self.retry(exc=exc)


# ============================================================================
# 내부 비동기 함수 (Celery Task에서 호출)
# ============================================================================

async def _save_chat_log_async(
    query: str,
    response: str,
    status: str,
    chat_model_id: int,
    input_time: datetime,
    output_time: datetime,
    latency_ms: int,
    category_id: Optional[int] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    guardrail_reason: Optional[str] = None
):
    """채팅 로그 저장 (비동기)
    
    독립된 DB 세션을 생성하여 사용 (Celery Worker는 별도 프로세스)
    """
    from app.infrastructure.persistence.session import async_db_session
    from app.domain.services.chat_log_service import ChatLogService
    
    async with async_db_session() as session:
        chat_log_service = ChatLogService(session)
        await chat_log_service.save_log(
            query=query,
            response=response,
            status=status,
            chat_model_id=chat_model_id,
            input_time=input_time,
            output_time=output_time,
            latency_ms=latency_ms,
            category_id=category_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            guardrail_reason=guardrail_reason
        )


@celery_app.task(
    name="cleanup_old_chat_logs",
    bind=True,
    max_retries=2,
    default_retry_delay=300,  # 5분 후 재시도
)
def cleanup_old_chat_logs_task(
    self,
    days_old: int = 90  # 기본값: 90일 (3개월)
):
    """오래된 채팅 로그 삭제 Celery Task
    
    Stability Fix: 오래된 로그(예: 3개월 이상)를 삭제하는 배치 작업
    Celery Beat를 사용하여 주기적으로 실행 (예: 매일 새벽 3시)
    
    Args:
        self: Celery Task 인스턴스 (bind=True)
        days_old: 삭제할 로그의 최소 일수 (기본값: 90일)
    
    Example (Celery Beat 설정):
        from celery.schedules import crontab
        celery_app.conf.beat_schedule = {
            'cleanup-old-logs': {
                'task': 'cleanup_old_chat_logs',
                'schedule': crontab(hour=3, minute=0),  # 매일 새벽 3시
            },
        }
    """
    try:
        from datetime import datetime, timedelta
        from app.infrastructure.persistence.session import db_session
        from sqlalchemy import delete
        from app.infrastructure.persistence.models import ChatLog
        
        # 삭제 기준 날짜 계산
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # 동기 DB 세션 사용
        with db_session() as db:
            # 오래된 로그 삭제 (soft delete: deleted_at 설정)
            # 또는 물리적 삭제: db.execute(delete(ChatLog).where(ChatLog.created_at < cutoff_date))
            
            # Soft delete (deleted_at 설정)
            deleted_count = db.query(ChatLog).filter(
                ChatLog.created_at < cutoff_date,
                ChatLog.deleted_at.is_(None)  # 이미 삭제된 것은 제외
            ).update(
                {"deleted_at": datetime.now()},
                synchronize_session=False
            )
            
            db.commit()
            
            logger.info(f"✅ [Celery] 오래된 채팅 로그 삭제 완료: {deleted_count}개 로그 삭제 (기준: {days_old}일 이전)")
            
    except Exception as exc:
        logger.error(f"⚠️ [Celery] 오래된 채팅 로그 삭제 실패: {exc}", exc_info=True)
        # 재시도 (최대 2회)
        raise self.retry(exc=exc)


@celery_app.task(
    name="generate_faqs_from_frequency",
    bind=True,
    max_retries=2,
    default_retry_delay=300,  # 5분 후 재시도
)
def generate_faqs_from_frequency_task(
    self,
    company_id: str = "default",
    days_back: int = 30,
    min_frequency: int = 10,
    min_confidence: float = 0.6, 
    cluster_similarity_threshold: float = 0.8
):
    """FAQ 빈도 기반 생성 Celery Task
    
    Fix: APScheduler에서 실행되던 작업을 Celery Beat로 이동
    매일 새벽 2시에 실행 (Celery Beat 스케줄 설정 필요)
    
    Args:
        self: Celery Task 인스턴스 (bind=True)
        company_id: 회사 ID (기본값: "default")
        days_back: 며칠 전까지의 데이터 조회 (기본값: 30일)
        min_frequency: 최소 빈도 (기본값: 20회)
        min_confidence: RAG 최소 신뢰도 (기본값: 0.6)
        cluster_similarity_threshold: 클러스터링 유사도 임계값 (기본값: 0.8)
    
    Example (Celery Beat 설정):
        from celery.schedules import crontab
        celery_app.conf.beat_schedule = {
            'generate-faqs-from-frequency': {
                'task': 'generate_faqs_from_frequency',
                'schedule': crontab(hour=2, minute=0),  # 매일 새벽 2시
            },
        }
    """
    try:
        from app.infrastructure.persistence.session import db_session
        from app.domain.services.faq_generation_service import FAQGenerationService
        
        from app.infrastructure.adapters.persistence.faq_repository_adapter import FAQRepositoryAdapter
        from app.infrastructure.adapters.cache.question_log_service import QuestionLogService
        from app.infrastructure.utils.redis_client import get_redis_client
        
        # 동기 DB 세션 사용
        with db_session() as db:
            faq_repo = FAQRepositoryAdapter(db)
            llm_provider = _get_llm_provider()
            similarity_checker = _get_similarity_checker()
            question_log_service = QuestionLogService(redis_client=get_redis_client())
            notification_service = _get_notification_service()
            
            faq_service = FAQGenerationService(
                faq_repository=faq_repo,
                llm_provider=llm_provider,
                similarity_checker=similarity_checker,
                question_log_service=question_log_service,
                notification_service=notification_service,
                db_session=db
            )
            
            # Fix: 메서드 이름 수정 및 async_to_sync 사용
            result = async_to_sync(faq_service.generate_faqs_from_frequency)(
                company_id=company_id,
                days_back=days_back,
                min_frequency=min_frequency,
                min_confidence=min_confidence,
                cluster_similarity_threshold=cluster_similarity_threshold
            )
            
            logger.info(
                f"✅ [Celery] FAQ 빈도 기반 생성 완료: "
                f"클러스터={result['total_clusters']}, "
                f"생성={result['faqs_created']}, "
                f"처리={result['clusters_processed']}"
            )
            
            if result['errors']:
                logger.warning(f"⚠️ [Celery] FAQ 생성 중 오류: {result['errors']}")
            
            return result
            
    except Exception as exc:
        logger.error(f"⚠️ [Celery] FAQ 빈도 기반 생성 실패: {exc}", exc_info=True)
        # 재시도 (최대 2회)
        raise self.retry(exc=exc)


# ============================================================================
# Dead Letter Queue (DLQ) 처리
# ============================================================================

def on_task_failure(self, exc, task_id, args, kwargs, einfo):
    """Celery 작업 실패 시 호출되는 콜백 (Dead Letter Queue)
    
    실패한 작업을 Redis List에 저장하여 나중에 검사하거나 재시도할 수 있게 함
    """
    try:
        from app.infrastructure.utils.redis_client import get_redis_client
        import json
        
        redis_client = get_redis_client()
        dlq_key = "celery_dead_letter_queue"
        
        failure_info = {
            "task_name": self.name,
            "task_id": task_id,
            "args": args,
            "kwargs": kwargs,
            "exception": str(exc),
            "traceback": str(einfo),
            "failed_at": datetime.now().isoformat()
        }
        
        # Redis List에 추가 (LPUSH)
        redis_client.lpush(dlq_key, json.dumps(failure_info, default=str))
        
        # 최대 개수 유지 (예: 1000개)
        redis_client.ltrim(dlq_key, 0, 999)
        
        logger.critical(f"💀 [DLQ] 작업이 최종 실패하여 DLQ에 저장됨: task={self.name}, id={task_id}")
        
    except Exception as e:
        logger.error(f"⚠️ [DLQ] DLQ 저장 실패: {e}", exc_info=True)

# Task에 on_failure 콜백 연결
save_chat_log_task.on_failure = on_task_failure
generate_faq_candidate_task.on_failure = on_task_failure
cleanup_old_chat_logs_task.on_failure = on_task_failure
generate_faqs_from_frequency_task.on_failure = on_task_failure
