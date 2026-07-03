"""RAG 서비스 포트 (인터페이스)"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List, Literal


class RAGPort(ABC):
    """RAG 서비스 포트 (인터페이스)
    
    RAG 검색 서비스의 추상 인터페이스입니다.
    모든 RAG 서비스 구현체는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def search(
        self,
        query: str,
        chat_history: Optional[List] = None,
        top_k: int = 5,
        use_claude: bool = True,
        settings: Optional[Dict] = None,
        confidence_method: Literal["reranker"] = "reranker"
    ) -> Tuple[bool, Optional[str], Optional[List[str]], float, Dict[int, str], List[str]]:
        """RAG 검색 및 답변 생성
        
        Args:
            query: 검색 쿼리 (사용자 질문)
            chat_history: 대화 히스토리 (선택적, 컨텍스트 제공)
            top_k: 검색할 문서 수 (기본값: 5)
            use_claude: Claude Sonnet 사용 여부 (기본값: True)
            settings: 추가 설정 (선택적)
            confidence_method: 신뢰도 계산 방식 (기본값: "reranker")
        
        Returns:
            (has_answer, answer, sources, confidence, doc_number_to_filename, related_queries) 튜플
            - has_answer: 답변 생성 여부 (bool)
            - answer: 생성된 답변 (str 또는 None)
            - sources: 문서 출처 리스트 (list 또는 None)
            - confidence: 신뢰도 점수 (0.0 ~ 1.0)
            - doc_number_to_filename: 문서 번호 → 파일명 매핑 (dict)
            - related_queries: 연관 검색어 리스트 (relevance_score >= 0.7인 문서들에서 추출)
        """
        pass

