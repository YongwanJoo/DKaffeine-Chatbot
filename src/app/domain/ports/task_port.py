"""Background Task 포트 인터페이스

Domain 레이어가 백그라운드 작업을 추상화하기 위한 포트입니다.
Infrastructure 레이어에서 이 포트를 구현합니다 (예: Celery).
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime


class BackgroundTaskPort(ABC):
    """백그라운드 작업 포트 인터페이스
    
    비동기 백그라운드 작업(Celery Task 등)을 추상화합니다.
    """
    
    @abstractmethod
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
        """채팅 로그 저장 작업을 백그라운드로 실행
        
        Args:
            query: 사용자 질문
            response: 챗봇 응답
            status: 로그 상태 (SUCCESS, GUARDRAIL, REQUERY, ERROR)
            chat_model_id: 챗봇 모델 ID
            input_time: 입력 시간
            output_time: 출력 시간
            latency_ms: 응답 지연 시간 (밀리초)
            category_id: 카테고리 ID (선택)
            prompt_tokens: 프롬프트 토큰 수 (선택)
            completion_tokens: 완료 토큰 수 (선택)
        """
        pass
    
    @abstractmethod
    def generate_faq_candidate(
        self,
        user_question: str,
        rag_answer: str,
        rag_confidence: float,
        chatbot_settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """FAQ 후보 생성 작업을 백그라운드로 실행
        
        Args:
            user_question: 사용자 질문
            rag_answer: RAG 답변
            rag_confidence: RAG 신뢰도
            chatbot_settings: 챗봇 설정 (선택)
        """
        pass
    
    @abstractmethod
    def cleanup_old_chat_logs(self, days_old: int = 90) -> None:
        """오래된 채팅 로그 삭제 작업을 백그라운드로 실행
        
        Args:
            days_old: 삭제할 로그의 최소 일수 (기본값: 90일)
        """
        pass
    
    @abstractmethod
    def generate_faqs_from_frequency(
        self,
        company_id: str = "default",
        days_back: int = 30,
        min_frequency: int = 20,
        min_confidence: float = 0.65,
        cluster_similarity_threshold: float = 0.7
    ) -> None:
        """빈도 기반 FAQ 생성 작업을 백그라운드로 실행
        
        Args:
            company_id: 회사 ID
            days_back: 며칠 전까지의 데이터 조회
            min_frequency: 최소 빈도
            min_confidence: RAG 최소 신뢰도
            cluster_similarity_threshold: 클러스터링 유사도 임계값
        """
        pass
