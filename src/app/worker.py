"""Celery Worker 설정

Redis를 메시지 브로커로 사용하는 Celery 백그라운드 작업 큐
"""
import logging
from celery import Celery
from app.infrastructure.config.config_loader import get_config

logger = logging.getLogger(__name__)

# Redis URL 구성
redis_url = get_config("url", None, "redis")

if not redis_url:
    redis_host = get_config("host", "localhost", "redis")
    redis_port = get_config("port", 6379, "redis")
    redis_db = get_config("db", 0, "redis")
    redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

logger.info(f"Celery Redis URL: {redis_url}")

# Celery 앱 생성
celery_app = Celery(
    "dkaffeine_chatbot",
    broker=redis_url,
    backend=redis_url,
    include=["app.infrastructure.jobs.tasks"]
)

# Celery 설정
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Fix: Celery Beat 스케줄 설정 (APScheduler 대체)
# 다중 워커 환경에서 안정적인 스케줄링 보장
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # 오래된 채팅 로그 삭제 (매일 새벽 3시)
    'cleanup-old-chat-logs': {
        'task': 'cleanup_old_chat_logs',
        'schedule': crontab(hour=3, minute=0),  # 매일 새벽 3시
        'options': {'queue': 'celery'}  # 기본 큐 사용
    },
    # FAQ 빈도 기반 생성 (테스트: 15시 10분)
    'generate-faqs-from-frequency': {
        'task': 'generate_faqs_from_frequency',
        'schedule': crontab(hour=15, minute=10),  # 테스트: 15시 10분
        'options': {'queue': 'celery'}  # 기본 큐 사용
    },
}

logger.info("Celery app initialized successfully")
logger.info("Celery Beat schedule configured: cleanup-old-chat-logs (03:00), generate-faqs-from-frequency (02:00)")
