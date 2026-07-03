"""FAQ 서비스 포트 (인터페이스)"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class FAQPort(ABC):
    """FAQ 서비스 포트 (인터페이스)
    
    FAQ 검색 서비스의 추상 인터페이스입니다.
    모든 FAQ 서비스 구현체는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def search(
        self,
        user_message: str,
        threshold: float = 0.7,
        user_embedding: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """FAQ 검색
        
        Args:
            user_message: 사용자 질문
            threshold: 유사도 임계값 (0.0 ~ 1.0, 기본값: 0.7)
            user_embedding: 사용자 질문 임베딩 (제공되면 재사용, None이면 생성)
        
        Returns:
            (faq_match, faq_answer, faq_confidence) 튜플
            - faq_match: 매칭 여부 (bool)
            - faq_answer: FAQ 답변 (str 또는 None)
            - faq_confidence: 유사도 점수 (float 또는 None)
        """
        pass

