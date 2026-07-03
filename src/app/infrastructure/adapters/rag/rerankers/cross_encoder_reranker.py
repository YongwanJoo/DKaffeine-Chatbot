"""Cross-Encoder 기반 Reranker

sentence-transformers의 Cross-Encoder를 사용하여 검색 결과를 재정렬합니다.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple, Optional
import numpy as np

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder 기반 Reranker
    
    sentence-transformers의 Cross-Encoder를 사용하여 검색 결과를 재정렬하고
    더 정확한 신뢰도 점수를 계산합니다.
    
    특징:
    - 질문-문서 쌍을 함께 고려하여 관련성 평가
    - 일반적으로 Bi-Encoder(코사인 유사도)보다 높은 점수 제공 (0.8~0.95)
    - 검색 결과 재정렬로 정확도 향상
    
    Example:
        ```python
        reranker = CrossEncoderReranker()
        reranked_docs = reranker.rerank(
            query="회사 소개",
            documents=[doc1, doc2, doc3],
            top_k=5
        )
        ```
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """CrossEncoderReranker 초기화
        
        Args:
            model_name: Cross-Encoder 모델 이름
                - "cross-encoder/ms-marco-MiniLM-L-6-v2": 빠르고 가벼움 (기본값)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2": 더 정확하지만 느림
            device: 사용할 디바이스 ("cpu", "cuda", None=자동)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialized = False
        
    def _initialize_model(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Cross-Encoder 모델 로딩 중: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            self._initialized = True
            logger.info(f"Cross-Encoder 모델 로딩 완료: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers 패키지가 설치되지 않았습니다.\\n"
                "설치: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Cross-Encoder 모델 로딩 실패: {e}")
            raise RuntimeError(f"Cross-Encoder 모델 초기화 실패: {e}") from e
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        scores_only: bool = False
    ) -> List[Tuple[Document, float]]:
        """검색 결과 재정렬 및 점수 재계산
        
        Args:
            query: 검색 쿼리
            documents: 재정렬할 문서 리스트
            top_k: 반환할 상위 문서 수 (None이면 모두 반환)
            scores_only: 점수만 반환할지 여부 (기본값: False, Document와 함께 반환)
            
        Returns:
            (Document, score) 튜플 리스트 (점수 내림차순 정렬)
            - Document: 재정렬된 문서
            - score: Reranker 점수 (일반적으로 0.0 ~ 1.0, 높을수록 관련성 높음)
        """
        if not documents:
            return []
        
        start_time = time.time()
        
        # 모델 초기화 (지연 로딩)
        self._initialize_model()
        
        # 질문-문서 쌍 생성
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 점수 계산
        try:
            raw_logits = self.model.predict(pairs)
            # raw_logits는 numpy array 또는 list
            if hasattr(raw_logits, 'tolist'):
                raw_logits = raw_logits.tolist()
            
            # Cross-Encoder는 raw logits를 반환 (보통 -10 ~ 10 범위)
            # Bedrock KB 점수 스케일(0.0~1.0)과 비교 가능하도록 변환
            raw_logits = np.array(raw_logits)
            
            # 표준 방법: Sigmoid 변환 사용
            # Cross-Encoder의 표준 사용법은 Sigmoid 변환입니다
            # Sigmoid는 logits를 0.0~1.0 범위의 확률값으로 변환합니다
            from scipy.special import expit  # Sigmoid 함수
            
            # Sigmoid 변환: 1 / (1 + exp(-x))
            # 결과는 0.0~1.0 범위이며, 일반적으로 0.5~0.95 범위의 점수를 제공합니다
            # - 높은 양수 logits (예: 5~10) → 0.99~1.0
            # - 중간 양수 logits (예: 2~5) → 0.88~0.99
            # - 낮은 양수 logits (예: 0~2) → 0.5~0.88
            # - 음수 logits (예: -10~0) → 0.0~0.5
            scores = expit(raw_logits)
            
            # Sigmoid 결과를 그대로 사용 (표준 방법)
            # 필요시 threshold를 조정하여 사용 (예: 0.7 또는 0.6)
            scores = scores.tolist()
        except Exception as e:
            logger.error(f"Reranker 점수 계산 실패: {e}")
            # 폴백: 모든 문서에 동일한 점수 부여
            scores = [0.5] * len(documents)
        
        # 점수와 문서를 함께 정렬 (내림차순)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # top_k만큼 반환
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]
        
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"CrossEncoderReranker 재정렬 완료: 쿼리='{query[:50]}...', "
            f"입력={len(documents)}개, 출력={len(doc_score_pairs)}개, "
            f"최고 점수={max(scores):.3f}, 최저 점수={min(scores):.3f}, "
            f"소요 시간={elapsed_time:.3f}초"
        )
        
        return doc_score_pairs
    
    def rerank_with_original_scores(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """원본 점수와 함께 재정렬 (비교용)
        
        Args:
            query: 검색 쿼리
            documents_with_scores: (Document, original_score) 튜플 리스트
            top_k: 반환할 상위 문서 수 (None이면 모두 반환)
            
        Returns:
            (Document, original_score, reranker_score) 튜플 리스트
            - Document: 재정렬된 문서
            - original_score: 원본 점수 (Bedrock KB 또는 하이브리드)
            - reranker_score: Reranker 점수
        """
        documents = [doc for doc, _ in documents_with_scores]
        original_scores = [score for _, score in documents_with_scores]
        
        # Reranker로 재정렬
        reranked = self.rerank(query, documents, top_k=top_k)
        
        # 원본 점수 매핑 (Document는 hashable하지 않으므로 인덱스 기반 매핑 사용)
        # Document의 page_content를 키로 사용 (내용이 같으면 같은 문서로 간주)
        doc_to_original_score = {}
        for doc, score in documents_with_scores:
            # page_content를 키로 사용 (내용 기반 매핑)
            doc_key = doc.page_content[:100] if doc.page_content else ""  # 처음 100자만 사용
            if doc_key not in doc_to_original_score:
                doc_to_original_score[doc_key] = score
        
        # 결과 생성
        results = []
        for doc, reranker_score in reranked:
            # 재정렬된 문서의 page_content로 원본 점수 찾기
            doc_key = doc.page_content[:100] if doc.page_content else ""
            original_score = doc_to_original_score.get(doc_key, 0.0)
            results.append((doc, original_score, reranker_score))
        
        return results

    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        scores_only: bool = False
    ) -> List[Tuple[Document, float]]:
        """비동기 검색 결과 재정렬"""
        import asyncio
        # CPU 바운드 작업이므로 스레드 풀에서 실행
        return await asyncio.to_thread(
            self.rerank, query, documents, top_k, scores_only
        )

    async def arerank_with_original_scores(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """비동기 원본 점수와 함께 재정렬"""
        import asyncio
        return await asyncio.to_thread(
            self.rerank_with_original_scores, query, documents_with_scores, top_k
        )
