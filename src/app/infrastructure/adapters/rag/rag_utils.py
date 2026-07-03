"""RAG 유틸리티 함수

RAG 서비스에서 사용하는 공통 유틸리티 함수들을 제공합니다.
"""
import re
import json
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


def extract_user_question(query: str) -> str:
    """멀티턴 쿼리에서 현재 사용자 질문만 추출
    
    Reranker는 짧고 명확한 질문에 최적화되어 있으므로,
    멀티턴 맥락이 포함된 쿼리에서 현재 질문만 추출합니다.
    
    Args:
        query: 멀티턴 맥락이 포함된 쿼리 (예: "대화 맥락:\n사용자: ...\n봇: ...\n\n현재 질문: 휴가 신청 방법")
    
    Returns:
        추출된 현재 질문 (예: "휴가 신청 방법")
    """
    # "현재 질문:" 이후의 텍스트 추출
    if "현재 질문:" in query:
        parts = query.split("현재 질문:")
        if len(parts) > 1:
            current_question = parts[-1].strip()
            logger.debug(f"멀티턴 쿼리에서 현재 질문 추출: {current_question}")
            return current_question
    
    # "Current question:" 이후의 텍스트 추출 (영어)
    if "Current question:" in query:
        parts = query.split("Current question:")
        if len(parts) > 1:
            current_question = parts[-1].strip()
            logger.debug(f"Extracted current question from multi-turn query: {current_question}")
            return current_question
    
    # 패턴이 없으면 원본 반환
    return query


def clean_markdown(text: str) -> str:
    """마크다운 문법 제거
    
    plain 포맷일 때 LLM이 생성한 마크다운 문법을 제거합니다.
    
    Args:
        text: 마크다운이 포함된 텍스트
    
    Returns:
        마크다운이 제거된 텍스트
    """
    # 이스케이프된 개행 문자를 실제 개행으로 변환
    text = text.replace('\\n', '\n')
    
    # 마크다운 문법 제거
    # 1. 굵게 (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # 2. 기울임 (*text* or _text_)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # 3. 코드 (`code`)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # 4. 제목 (# Heading)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # 5. 리스트 (- item or * item)
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    
    # 6. 링크 ([text](url))
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    return text


def parse_json_response(
    raw_response: str,
    response_format: str
) -> Tuple[str, List[int], List[str]]:
    """LLM 응답을 JSON으로 파싱하고 마크다운 제거
    
    Args:
        raw_response: LLM 원본 응답
        response_format: "plain" 또는 "markdown"
    
    Returns:
        (answer, document_numbers, related_queries)
    """
    # JSON 추출 (코드 블록 제거)
    json_str = raw_response.strip()
    
    # news_usecase.py와 동일한 로직 적용 (더 강력한 마크다운 제거)
    if "```" in json_str:
        json_str = json_str.replace("```json", "").replace("```", "").strip()
    
    # JSON 객체 찾기 (첫 번째 { 부터 마지막 } 까지)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(json_str):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = json_str[start_idx:i+1]
                break
    
    # JSON 파싱 시도
    try:
        parsed = json.loads(json_str, strict=False)
        answer = parsed.get("answer", "").strip()
        document_numbers = parsed.get("document_numbers", [])
        related_queries = parsed.get("related_queries", [])
        
        if not answer:
            logger.warning("JSON 파싱 성공했지만 answer가 비어있음. 원본 응답 사용.")
            answer = raw_response
            document_numbers = []
            related_queries = []
        else:
            logger.debug(f"JSON 파싱 성공: answer 길이={len(answer)}, document_numbers={document_numbers}")
            
            # plain 포맷일 때 마크다운 문법 제거
            if response_format != "markdown":
                answer = clean_markdown(answer)
                logger.debug(f"마크다운 제거 완료: answer 길이={len(answer)}")
                
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 파싱 실패: {e}. 원본 응답을 텍스트로 처리합니다.")
        logger.debug(f"파싱 시도한 JSON (첫 500자): {json_str[:500]}")
        
        # Fallback: Regex로 answer 필드 추출 시도
        # 1. 정상적인 "answer": "..." 패턴 찾기 (이스케이프 문자 처리 포함)
        # DOTALL 플래그로 개행 문자 포함
        answer_match = re.search(r'"answer"\s*:\s*"(.*?)(?<!\\)"', json_str, re.DOTALL)
        
        # 2. 잘린 JSON 처리: "answer": "..." 패턴이 없으면, "answer": " 이후의 모든 텍스트 추출
        if not answer_match:
            # answer 필드가 닫히지 않고 끝난 경우 (Truncated JSON)
            trunc_match = re.search(r'"answer"\s*:\s*"(.*)', json_str, re.DOTALL)
            if trunc_match:
                logger.warning("잘린 JSON 감지 (answer 필드 미종료). answer 필드 부분 추출 시도.")
                answer_match = trunc_match
        
        if answer_match:
            answer = answer_match.group(1)
            # 이스케이프된 문자 복원 (순서 중요: 먼저 복잡한 것부터)
            # \\ -> \ (백슬래시 먼저 처리)
            answer = answer.replace('\\\\', '\\')
            # \n -> 줄바꿈
            answer = answer.replace('\\n', '\n')
            # \r -> 캐리지 리턴
            answer = answer.replace('\\r', '\r')
            # \t -> 탭
            answer = answer.replace('\\t', '\t')
            # \" -> "
            answer = answer.replace('\\"', '"')
            # \' -> ' (JSON에서는 사용 안 하지만 안전을 위해)
            answer = answer.replace("\\'", "'")
            
            # 잘린 JSON인 경우 경고 추가
            if not json_str.rstrip().endswith('}'):
                logger.warning(
                    f"⚠️ 잘린 JSON 응답 감지: max_tokens가 부족할 수 있습니다. "
                    f"답변 길이={len(answer)}자. DB 설정에서 max_tokens를 2000 이상으로 권장합니다."
                )
            
            logger.info(f"Regex로 answer 필드 추출 성공: 길이={len(answer)}자")
            
            # plain 포맷일 때 마크다운 제거
            if response_format != "markdown":
                answer = clean_markdown(answer)
        else:
            # answer 필드도 못 찾으면 원본 사용하되, JSON 형식처럼 보이면 정제 시도
            if '"answer":' in raw_response:
                logger.warning("Regex 추출 실패, JSON 구조가 깨진 것으로 보임. 원본 반환.")
            answer = raw_response
            
        document_numbers = []
        related_queries = []
    
    return answer, document_numbers, related_queries


def extract_filename_from_uri(uri: str) -> str:
    """URI에서 파일명 추출 (메타데이터가 없을 때 폴백용)
    
    다양한 URI 형식을 지원:
    - s3://bucket/path/file.pdf -> file.pdf
    - s3://bucket/path/file.pdf/paragraphs/v1/xxx.json -> file.pdf
    - file:///path/to/document.pdf -> document.pdf
    - /path/to/document.pdf -> document.pdf
    - http://example.com/file.pdf -> file.pdf
    
    Args:
        uri: 파일 URI 또는 경로
    
    Returns:
        추출된 파일명 또는 원본 URI (추출 실패 시)
    """
    if not uri or not isinstance(uri, str):
        return uri or "unknown"
    
    # 경로 분리
    parts = uri.split("/")
    
    # 파일명 찾기 (확장자가 있는 부분)
    # 지원하는 확장자 목록
    supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".xlsx", ".pptx"]
    
    # 역순으로 검색 (마지막에 있는 파일명 우선)
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        if "." in part and any(part.lower().endswith(ext) for ext in supported_extensions):
            return part
    
    # 파일명을 찾지 못하면 마지막 의미있는 부분 반환
    if len(parts) > 0:
        last_part = parts[-1]
        if last_part and last_part != "":
            return last_part
    
    # 그래도 못 찾으면 원본 반환
    return uri
