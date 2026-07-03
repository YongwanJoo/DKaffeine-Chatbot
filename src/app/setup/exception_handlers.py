"""FastAPI 예외 핸들러
"""
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.domain.exceptions import ChatbotException

logger = logging.getLogger(__name__)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 오류 핸들러 (422 오류 상세 로깅)"""
    import json
    
    # 검증 오류 상세 정보 수집
    errors = exc.errors()
    
    logger.warning(f"❌❌❌ 요청 검증 실패 (422)")
    logger.warning(f"❌❌❌ 요청 경로: {request.url.path}")
    logger.warning(f"❌❌❌ 요청 메서드: {request.method}")
    logger.warning(f"❌❌❌ 검증 오류 개수: {len(errors)}")
    
    # 각 오류 상세 출력
    for i, error in enumerate(errors, 1):
        logger.warning(f"❌❌❌ 오류 #{i}:")
        logger.warning(f"   위치: {error.get('loc', [])}")
        logger.warning(f"   메시지: {error.get('msg', '')}")
        logger.warning(f"   타입: {error.get('type', '')}")
        if error.get('ctx'):
            logger.warning(f"   컨텍스트: {error.get('ctx', {})}")
    
    # 요청 본문 로깅 (보안: 민감 정보 마스킹)
    if hasattr(exc, 'body') and exc.body:
        try:
            if isinstance(exc.body, dict):
                # 민감한 필드 마스킹
                masked_body = {}
                sensitive_fields = ['message', 'chat_history', 'answer', 'content', 'user_message']
                for key, value in exc.body.items():
                    if key in sensitive_fields:
                        if isinstance(value, str):
                            masked_body[key] = f"[마스킹됨: 길이={len(value)}자]"
                        elif isinstance(value, list):
                            masked_body[key] = f"[마스킹됨: 리스트 길이={len(value)}]"
                        else:
                            masked_body[key] = "[마스킹됨]"
                    else:
                        masked_body[key] = value
                
                body_summary = json.dumps(masked_body, ensure_ascii=False, indent=2)
                logger.warning(f"❌❌❌ 요청 본문 (요약, 민감 정보 마스킹):")
                logger.warning(f"   필드: {list(exc.body.keys())}")
                logger.warning(f"   본문 (마스킹):\n{body_summary[:1000]}")
            elif isinstance(exc.body, list):
                logger.warning(f"❌❌❌ 요청 본문 (요약): 리스트 길이={len(exc.body)}")
            else:
                body_str = str(exc.body)
                logger.warning(f"❌❌❌ 요청 본문 (요약): 길이={len(body_str)}자, 타입={type(exc.body).__name__}")
        except Exception as e:
            logger.warning(f"❌❌❌ 요청 본문 파싱 실패: {e}")
            # 파싱 실패 시에도 전체 내용을 로깅하지 않음
            if hasattr(exc, 'body'):
                body_str = str(exc.body)
                logger.warning(f"❌❌❌ 요청 본문 (요약): 길이={len(body_str)}자, 파싱 실패")
    
    # 응답에도 민감 정보 포함하지 않음
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": errors,
            # body_preview 제거 (보안: 민감 정보 유출 방지)
        }
    )

async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    """챗봇 커스텀 예외 핸들러
    
    모든 ChatbotException 계열 예외를 일관된 형식으로 처리합니다.
    """
    logger.error(
        f"챗봇 예외 발생: {exc.__class__.__name__} - {exc.message} "
        f"(status_code={exc.status_code}, error_code={exc.error_code})"
    )
    
    if exc.details:
        logger.debug(f"예외 상세 정보: {exc.details}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "error_type": exc.__class__.__name__,
            "details": exc.details
        }
    )
