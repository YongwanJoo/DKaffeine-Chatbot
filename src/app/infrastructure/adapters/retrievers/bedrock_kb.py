"""Bedrock Knowledge Base Retriever"""
from __future__ import annotations

import os
import logging
from typing import List, Tuple, Optional, Any

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from langchain_core.documents import Document

from app.infrastructure.config.config_loader import (
    get_bedrock_config,
    get_config_int,
    get_config_float
)
from app.infrastructure.utils.resilience import CircuitBreaker, retry_with_backoff
from . import BaseRetriever

logger = logging.getLogger(__name__)


class BedrockKBRetriever(BaseRetriever):
    """AWS Bedrock Knowledge Base 검색기
    
    Bedrock KB의 Retrieve API를 사용하여 문서를 검색합니다.
    """

    def __init__(
        self,
        region: str | None = None,
        knowledge_base_id: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        redis_client: Any | None = None,
    ):
        """BedrockKBRetriever 초기화
        
        Args:
            region: AWS 리전 (예: "ap-northeast-1")
            knowledge_base_id: Bedrock Knowledge Base ID
            aws_access_key_id: AWS Access Key ID (선택적, 환경변수 사용 가능)
            aws_secret_access_key: AWS Secret Access Key (선택적, 환경변수 사용 가능)
        """
        # 설정 로드 (환경변수 또는 .secrets.toml)
        config = get_bedrock_config()
        
        # Security Fix: 하드코딩된 리전 제거, 설정 없을 경우 ValueError
        self._region = region or config.get("region")
        if not self._region:
            raise ValueError("AWS 리전이 설정되지 않았습니다. 환경변수 BEDROCK_REGION 또는 .secrets.toml의 [bedrock].region을 설정하세요.")
        self._kb_id = knowledge_base_id or config.get("knowledge_base_id")
        self._access_key_id = aws_access_key_id or config.get("access_key_id")
        self._secret_access_key = aws_secret_access_key or config.get("secret_access_key")
        
        if not self._kb_id:
            raise ValueError(
                "Bedrock Knowledge Base ID가 설정되지 않았습니다.\n"
                "환경변수 BEDROCK_KB_ID 또는 .secrets.toml의 [bedrock].knowledge_base_id를 설정하세요."
            )
        
        # boto3 세션 생성
        if self._access_key_id and self._secret_access_key:
            session = boto3.session.Session(
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name=self._region,
            )
            logger.info(f"BedrockKBRetriever: 자격증명 사용 (access_key_id={self._access_key_id[:10]}...)")
        else:
            session = boto3.session.Session(region_name=self._region)
            logger.info("BedrockKBRetriever: 환경변수/프로파일 자격증명 사용")
        
        # bedrock-agent-runtime 클라이언트 생성 (검색용)
        self._client = session.client("bedrock-agent-runtime", region_name=self._region)
        
        # Circuit Breaker 초기화
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=get_config_int("BEDROCK_CIRCUIT_BREAKER_THRESHOLD", 5, section="bedrock_circuit_breaker"),
            recovery_timeout=get_config_float("BEDROCK_CIRCUIT_BREAKER_TIMEOUT", 60.0, section="bedrock_circuit_breaker"),
            expected_exception=(ClientError, BotoCoreError),
            name="bedrock_kb",
            redis_client=redis_client
        )
        
        logger.info(
            f"BedrockKBRetriever 초기화 완료: "
            f"KB ID={self._kb_id}, 리전={self._region}"
        )

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Bedrock KB에서 유사 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수 (최대 10)
            
        Returns:
            (Document, score) 튜플 리스트
            - Document: 검색된 문서 (page_content, metadata 포함)
            - score: 유사도 점수 (0.0 ~ 1.0, 높을수록 유사)
            
        Raises:
            ClientError: AWS API 호출 실패 시
            ValueError: KB ID가 설정되지 않았을 때
        """
        if k > 10:
            k = 10  # Bedrock KB는 최대 10개까지 반환
            logger.warning(f"검색 개수가 10개를 초과하여 10개로 제한합니다.")
        
        # Retry 적용된 검색 함수
        @retry_with_backoff(
            max_retries=get_config_int("BEDROCK_MAX_RETRIES", 3, section="bedrock_circuit_breaker"),
            initial_delay=get_config_float("BEDROCK_RETRY_INITIAL_DELAY", 1.0, section="bedrock_circuit_breaker"),
            max_delay=get_config_float("BEDROCK_RETRY_MAX_DELAY", 60.0, section="bedrock_circuit_breaker"),
            retry_on=(ClientError, BotoCoreError),
            on_retry=lambda e, attempt: logger.warning(
                f"Bedrock KB search failed (attempt {attempt}): {e}. Retrying..."
            )
        )
        def _retrieve_documents():
            return self._client.retrieve(
                knowledgeBaseId=self._kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": k
                    }
                }
            )
        
        try:
            # Circuit Breaker를 통해 검색 실행
            response = self.circuit_breaker.call(_retrieve_documents)
            
            documents: List[Tuple[Document, float]] = []
            
            for result in response.get("retrievalResults", []):
                # 문서 내용 추출
                content = result.get("content", {}).get("text", "")
                if not content:
                    continue
                
                # 유사도 점수 추출 (0.0 ~ 1.0, 높을수록 유사)
                score = result.get("score", 0.0)
                
                # 메타데이터 추출
                metadata = result.get("metadata", {})
                location = result.get("location", {})
                
                # S3 URI 추출
                s3_location = location.get("s3Location", {})
                uri = s3_location.get("uri", "")
                
                # 정제된 파일명 추출 (메타데이터 우선, 없으면 URI에서 추출)
                # Bedrock KB의 메타데이터에 title, filename 등이 있을 수 있음
                display_name = (
                    metadata.get("title") or 
                    metadata.get("filename") or 
                    metadata.get("name") or 
                    None
                )
                
                # Document 생성
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": uri or metadata.get("source", "unknown"),
                        "chunk_id": result.get("location", {}).get("type", ""),
                        "score": score,
                        "display_name": display_name,  # 정제된 파일명/제목 (메타데이터에서 추출)
                        **{k: v for k, v in metadata.items() if k != "source"}
                    }
                )
                
                # Bedrock KB는 유사도 점수(0.0~1.0, 높을수록 유사)를 반환하므로 그대로 사용
                # FAISS는 사용하지 않으므로 변환 불필요
                documents.append((doc, score))
            
            logger.info(
                f"Bedrock KB 검색 완료: 쿼리='{query[:50]}...', "
                f"요청={k}개, 반환={len(documents)}개, 최고 점수={max([s for _, s in documents], default=0.0):.3f}"
            )
            
            return documents
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            
            logger.error(
                f"Bedrock KB 검색 실패: {error_code} - {error_msg}\n"
                f"KB ID={self._kb_id}, 리전={self._region}"
            )
            
            # 권한 오류 또는 KB를 찾을 수 없는 경우
            if error_code in ["ResourceNotFoundException", "AccessDeniedException"]:
                raise ValueError(
                    f"Bedrock Knowledge Base 접근 실패: {error_msg}\n"
                    f"KB ID={self._kb_id}, 리전={self._region}\n"
                    f"자격증명 및 권한을 확인하세요."
                ) from e
            
            raise
        except Exception as e:
            logger.error(f"Bedrock KB 검색 중 예상치 못한 오류: {e}")
            raise

    async def asimilarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Bedrock KB에서 유사 문서 비동기 검색
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수 (최대 10)
            
        Returns:
            (Document, score) 튜플 리스트
        """
        import asyncio
        # boto3는 동기 라이브러리이므로 스레드 풀에서 실행하여 비동기 처리
        return await asyncio.to_thread(self.similarity_search_with_score, query, k)
