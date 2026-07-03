"""FastAPI 의존성 주입 함수들

FastAPI의 Depends를 활용하여 서비스를 의존성으로 주입합니다.
테스트 시 모킹이 쉬워지고, 전역 상태 의존을 제거합니다.
"""
from fastapi import Request, HTTPException, status
from typing import Dict, Any
import logging

from app.domain.ports import (
    GuardrailPort,
    CachePort,
    FAQPort,
    RAGPort,
)

logger = logging.getLogger(__name__)


def get_services(request: Request) -> Dict[str, Any]:
    """
    app.state.services에서 모든 서비스를 가져옵니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        서비스 딕셔너리
        
    Raises:
        HTTPException: 서비스가 없을 때 500 에러
    """
    services = getattr(request.app.state, "services", None)
    if not services:
        logger.error("app.state.services가 없습니다.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서비스가 초기화되지 않았습니다. 서버를 재시작해주세요."
        )
    return services


def get_guardrail_service(request: Request) -> GuardrailPort:
    """
    Guardrail 서비스를 가져옵니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        GuardrailPort 인스턴스
        
    Raises:
        HTTPException: 서비스가 없을 때 500 에러
    """
    services = get_services(request)
    guardrail_service = services.get("guardrail_service")
    if not guardrail_service:
        logger.error(f"guardrail_service가 없습니다. services 키: {list(services.keys())}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Guardrail 서비스를 가져올 수 없습니다."
        )
    return guardrail_service


def get_cache_service(request: Request) -> CachePort:
    """
    Cache 서비스를 가져옵니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        CachePort 인스턴스
        
    Raises:
        HTTPException: 서비스가 없을 때 500 에러
    """
    services = get_services(request)
    cache_service = services.get("cache_service")
    if not cache_service:
        logger.error(f"cache_service가 없습니다. services 키: {list(services.keys())}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache 서비스를 가져올 수 없습니다."
        )
    return cache_service


def get_faq_service(request: Request) -> FAQPort:
    """
    FAQ 서비스를 가져옵니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        FAQPort 인스턴스
        
    Raises:
        HTTPException: 서비스가 없을 때 500 에러
    """
    services = get_services(request)
    faq_service = services.get("faq_service")
    if not faq_service:
        logger.error(f"faq_service가 없습니다. services 키: {list(services.keys())}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FAQ 서비스를 가져올 수 없습니다."
        )
    return faq_service


def get_rag_service(request: Request) -> RAGPort:
    """
    RAG 서비스를 가져옵니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        RAGPort 인스턴스
        
    Raises:
        HTTPException: 서비스가 없을 때 500 에러
    """
    services = get_services(request)
    rag_service = services.get("rag_service")
    if not rag_service:
        logger.error(f"rag_service가 없습니다. services 키: {list(services.keys())}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG 서비스를 가져올 수 없습니다."
        )
    return rag_service


def get_llm_provider(request: Request):
    """
    LLM Provider를 가져옵니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        LLMProvider 인스턴스
        
    Raises:
        HTTPException: 서비스가 없을 때 500 에러
    """
    services = get_services(request)
    provider = services.get("provider")
    if not provider:
        logger.error(f"provider가 없습니다. services 키: {list(services.keys())}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM Provider를 가져올 수 없습니다."
        )
    return provider





def get_all_services(request: Request) -> Dict[str, Any]:
    """
    모든 서비스를 딕셔너리로 반환합니다.
    LangGraph config로 사용하기 위한 헬퍼 함수입니다.
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        서비스 딕셔너리 (LangGraph config 형식)
        
    Raises:
        HTTPException: 필수 서비스가 없을 때 500 에러
    """
    services = get_services(request)
    
    # 필수 서비스 확인
    required_services = ["guardrail_service", "cache_service", "faq_service", "rag_service", "provider", "news_service"]
    missing_services = [svc for svc in required_services if not services.get(svc)]
    if missing_services:
        logger.error(f"필수 서비스가 누락되었습니다: {missing_services}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"필수 서비스가 누락되었습니다: {missing_services}"
        )
    
    # LangGraph config 형식으로 반환
    return {
        "guardrail_service": services.get("guardrail_service"),
        "cache_service": services.get("cache_service"),
        "faq_service": services.get("faq_service"),
        "rag_service": services.get("rag_service"),
        "news_service": services.get("news_service"),
        "provider": services.get("provider"),
    }

