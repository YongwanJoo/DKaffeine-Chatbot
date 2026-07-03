"""Celery Task 어댑터

Domain의 BackgroundTaskPort를 Celery로 구현한 어댑터입니다.
"""
from datetime import datetime
from typing import Optional, Dict, Any

from app.domain.constants import FAQ_MIN_CONFIDENCE, FAQ_MIN_FREQUENCY, FAQ_CLUSTER_SIMILARITY
from app.domain.ports.task_port import BackgroundTaskPort
from app.infrastructure.jobs.tasks import (
    save_chat_log_task,
    generate_faq_candidate_task,
    cleanup_old_chat_logs_task,
    generate_faqs_from_frequency_task
)


class CeleryTaskAdapter(BackgroundTaskPort):
    """Celery를 사용한 BackgroundTask 포트 구현
    
    Domain 레이어가 Celery에 직접 의존하지 않도록 어댑터 패턴을 적용합니다.
    """
    
    def save_chat_log(
        self,
        query: str,
        response: str,
        status: str,
        chat_model_id: int,
        input_time: datetime,
        output_time: datetime,
        latency_ms: int,
        category_id: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None
    ) -> None:
        """채팅 로그 저장 작업을 백그라운드로 실행"""
        save_chat_log_task.delay(
            query=query,
            response=response,
            status=status,
            chat_model_id=chat_model_id,
            input_time_iso=input_time.isoformat(),
            output_time_iso=output_time.isoformat(),
            latency_ms=latency_ms,
            category_id=category_id,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0
        )
    
    def generate_faq_candidate(
        self,
        user_question: str,
        rag_answer: str,
        rag_confidence: float,
        chatbot_settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """FAQ 후보 생성 작업을 백그라운드로 실행"""
        generate_faq_candidate_task.delay(
            user_question=user_question,
            rag_answer=rag_answer,
            rag_confidence=rag_confidence,
            chatbot_settings=chatbot_settings
        )
    
    def cleanup_old_chat_logs(self, days_old: int = 90) -> None:
        """오래된 채팅 로그 삭제 작업을 백그라운드로 실행"""
        cleanup_old_chat_logs_task.delay(days_old=days_old)
    
    def generate_faqs_from_frequency(
        self,
        company_id: str = "default",
        days_back: int = 30,
        min_frequency: int = FAQ_MIN_FREQUENCY,
        min_confidence: float = FAQ_MIN_CONFIDENCE,
        cluster_similarity_threshold: float = FAQ_CLUSTER_SIMILARITY
    ) -> None:
        """빈도 기반 FAQ 생성 작업을 백그라운드로 실행"""
        generate_faqs_from_frequency_task.delay(
            company_id=company_id,
            days_back=days_back,
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            cluster_similarity_threshold=cluster_similarity_threshold
        )
