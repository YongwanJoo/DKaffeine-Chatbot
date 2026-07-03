"""Domain Services - 상태 관리 유틸리티"""
from typing import Optional
from datetime import datetime
import uuid

from app.domain.ports.config_port import ConfigPort


def _load_guardrail_config(config_port: Optional[ConfigPort] = None) -> dict:
    """Guardrail 설정 파일 로드 (파일 시스템 우선)
    
    Args:
        config_port: 설정 포트 (선택적, 없으면 기본 어댑터 사용)
    """
    if config_port is None:
        # 기본 어댑터 사용 (순환 참조 방지를 위해 지연 import)
        from app.infrastructure.adapters.config.file_config_adapter import FileConfigAdapter
        config_port = FileConfigAdapter()
    
    return config_port.load_file("guardrail_config.json")


def create_initial_state(
    user_message: str,
    user_id: str,
    session_id: Optional[str] = None,
    chat_history: Optional[list] = None,
    top_p: Optional[float] = None,
    config_port: Optional[ConfigPort] = None,
    chatbot_settings: Optional[dict] = None,
    is_requery: bool = False
) -> dict:
    """초기 상태 생성
    
    Args:
        user_message: 사용자 메시지
        user_id: 사용자 ID
        session_id: 세션 ID (선택)
        chat_history: 대화 히스토리 (선택)
        top_p: 신뢰도 임계값 (선택, 없으면 설정에서 가져옴)
        chatbot_settings: 외부에서 주입된 챗봇 설정 (선택, 있으면 우선 사용)
    """
    # 챗봇 설정 조회 (chatbot_settings가 주입되면 사용, 없으면 기본값)
    if chatbot_settings:
        # 외부에서 주입된 설정 사용 (공유 DB 등)
        settings_dict = chatbot_settings
        # 필요한 값 추출 (기본값 처리)
        final_top_p = top_p if top_p is not None else settings_dict.get("top_p", 0.8)
    else:
        # 기본값 사용 (chatbot_settings가 없으면 기본값으로 초기화)
        # 기본 모델을 claude-haiku-4-5로 변경 (응답 속도 최적화)
        settings_dict = {
            "llm_model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.8,
            "search_results_count": 5,
            "persona_type": "professional",
            "persona_description": "",
            "response_length": "medium",
            "guardrail_keywords": {},  # 딕셔너리여야 함 (리스트 아님)
            "thresholds": {},
        }
        final_top_p = top_p if top_p is not None else 0.8
    
    return {
        "user_message": user_message,
        "user_id": user_id,
        "session_id": session_id or str(uuid.uuid4()),
        "chat_history": chat_history or [],
        "blacklist_blocked": False,
        "blacklist_category": None,
        "blacklist_reason": None,
        "blacklist_matched_patterns": None,
        "guardrail_passed": False,
        "guardrail_reason": None,
        "guardrail_confidence": None,
        "guardrail_check_details": None,
        "intent_type": None,
        "intent_category": None,
        "intent_confidence": None,
        "intent_analysis_details": None,
        "model_used": None,
        "cache_key": None,
        "cache_hit": False,
        "cached_response": None,
        "faq_match": False,
        "faq_answer": None,
        "faq_confidence": None,
        "answer_available": False,
        "rag_answer": None,
        "rag_sources": None,
        "rag_confidence": None,
        "has_answer": False,
        "related_queries": [],
        "confidence_passed": False,
        "top_p": final_top_p,
        "chatbot_settings": settings_dict,
        "needs_rerun": False,
        "is_requery": is_requery,
        "previous_questions": [],
        "token_usage": {},
        "final_answer": None,
        "final_sources": None,
        "final_message": None,
        "blocked": False,
        "block_reason": None,
        "faq_count": 0,
        "route": "start",
        "faq_generated": None,
        "faq_generation_message": None
    }

def _extract_filename_from_uri(uri: str) -> str:
    """URI에서 파일명 추출 (환경 독립적, 다양한 URI 형식 지원)
    
    주의: 이 함수는 rag.py에서 이미 정제된 파일명을 전달받으므로
    대부분의 경우 그대로 반환합니다. 폴백으로만 사용됩니다.
    
    예시:
    - s3://bucket/docs/file.pdf/paragraphs/v1/xxx.json -> file.pdf
    - s3://bucket/path/to/document.pdf -> document.pdf
    - file:///path/to/document.pdf -> document.pdf
    - /path/to/document.pdf -> document.pdf
    
    Args:
        uri: 파일 URI 또는 경로 (또는 이미 정제된 파일명)
    
    Returns:
        추출된 파일명 또는 원본 (추출 실패 시)
    """
    if not uri or not isinstance(uri, str):
        return uri or "unknown"
    
    # 이미 정제된 파일명인 경우 (확장자가 있고 경로 구분자가 없음)
    if "." in uri and "/" not in uri and not uri.startswith(("s3://", "file://", "http://", "https://")):
        return uri
    
    # 경로 분리
    parts = uri.split("/")
    
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


def format_answer_markdown(answer: str) -> str:
    """답변을 Markdown 형식으로 포맷팅
    
    모든 답변(casual, RAG, FAQ)에 대해 일관된 Markdown 형식을 적용합니다.
    줄바꿈, 리스트, 볼드 등을 적절히 처리합니다.
    
    Args:
        answer: 원본 답변 텍스트
        
    Returns:
        Markdown 형식으로 포맷팅된 답변 텍스트
    """
    if not answer:
        return answer
    
    import re
    
    # 1. 기본 텍스트 정리 (연속된 공백 정리, 줄바꿈 정리)
    # 연속된 공백을 하나로
    answer = re.sub(r' +', ' ', answer)
    # 줄 끝 공백 제거
    answer = '\n'.join(line.rstrip() for line in answer.split('\n'))
    
    # 2. 문장 끝 표시 뒤 줄바꿈 추가 (마침표, 물음표, 느낌표)
    # 문장 끝 + 공백 + 대문자/한글 시작 -> 줄바꿈 추가
    answer = re.sub(r'([.!?])\s+([A-Z가-힣])', r'\1\n\n\2', answer)
    # 이모지 뒤에도 줄바꿈 추가 (이모지 + 공백 + 대문자/한글)
    # 이모지 유니코드 범위: U+1F300-U+1F9FF, U+2600-U+26FF, U+2700-U+27BF
    emoji_pattern = r'[\U0001F300-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]'
    answer = re.sub(f'({emoji_pattern})\\s+([A-Z가-힣])', r'\1\n\n\2', answer)
    
    # 3. 번호 목록 정리 (1. 2. 3. 형식) - 각 항목 사이에 빈 줄 추가
    lines = answer.split('\n')
    formatted_lines = []
    in_list = False
    prev_was_list = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 번호 목록 감지 (1. 2. 3. 또는 1) 2) 3) 형식)
        list_match = re.match(r'^(\d+)[.)]\s+(.+)$', stripped)
        if list_match:
            if not in_list:
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')  # 리스트 전에 빈 줄 추가
                in_list = True
            if prev_was_list:
                formatted_lines.append('')  # 이전 항목과 사이에 빈 줄 추가
            number = list_match.group(1)
            content = list_match.group(2)
            formatted_lines.append(f"{number}. {content}")
            prev_was_list = True
        else:
            if in_list and stripped:
                formatted_lines.append('')  # 리스트 후에 빈 줄 추가
                in_list = False
            prev_was_list = False
            formatted_lines.append(line)
    
    answer = '\n'.join(formatted_lines)
    
    # 4. 콜론 뒤 줄바꿈 추가 (항목 구분)
    # "**제목:** 내용" 형식에서 콜론 뒤 줄바꿈
    # 단, URL(http://, https://)인 경우는 제외
    # Negative lookahead를 사용하여 http/https가 아닌 경우만 매칭
    answer = re.sub(r':\s+(?!http)([^:\n])', r':\n\1', answer)
    
    # 시간 형식 복구 (HH:MM)
    answer = re.sub(r'(\d{1,2}):\n(\d{2})', r'\1:\2', answer)
    
    # 5. 하이픈(-)으로 시작하는 하위 항목 처리
    # "- 항목" 형식을 Markdown 리스트로 변환하고, 앞에 줄바꿈 추가
    lines = answer.split('\n')
    formatted_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 하이픈으로 시작하는 항목 감지 (리스트 항목)
        if re.match(r'^-\s+', stripped):
            # 이전 줄이 비어있지 않으면 빈 줄 추가
            if formatted_lines and formatted_lines[-1].strip():
                formatted_lines.append('')
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    answer = '\n'.join(formatted_lines)
    
    # 6. 최종 정리: 연속된 줄바꿈을 2개로 제한
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    # 줄 끝 공백 제거
    answer = '\n'.join(line.rstrip() for line in answer.split('\n'))
    
    return answer.strip()


def format_rag_answer(answer: str) -> str:
    """RAG 답변을 읽기 쉬운 형식으로 포맷팅 (하위 호환성)
    
    이 함수는 format_answer_markdown을 호출합니다.
    """
    return format_answer_markdown(answer)


def format_response(state: dict) -> dict:
    """최종 응답 포맷팅 (프레젠테이션 계층)
    
    원본 데이터를 사용자에게 보여줄 형식으로 변환합니다.
    서식 처리는 이 함수에서만 수행되며, 캐시에는 원본 데이터만 저장됩니다.
    """
    # 출처 정보 처리 (RAG 출처는 파일명만 추출)
    sources = state.get("final_sources")
    if sources and isinstance(sources, list):
        # FAQ가 아닌 경우 파일명만 추출
        formatted_sources = []
        for source in sources:
            if source == "FAQ":
                formatted_sources.append("FAQ")
            else:
                # URI에서 파일명 추출 (rag.py에서 이미 정제된 파일명일 수 있음)
                filename = _extract_filename_from_uri(source)
                formatted_sources.append(filename)
        sources = formatted_sources
    else:
        # sources가 None이거나 리스트가 아니면 빈 리스트로 설정
        sources = []
    
    # 답변 가져오기 (원본 데이터)
    # final_answer가 없으면 rag_answer, cached_response, faq_answer 순서로 시도
    answer = (
        state.get("final_answer") or 
        state.get("final_message") or 
        state.get("rag_answer") or 
        state.get("cached_response") or 
        state.get("faq_answer")
    )
    
    # 모든 답변에 대해 Markdown 포맷팅 적용
    if answer:
        answer = format_answer_markdown(answer)
        
        # RAG 답변인 경우 참고 문서 섹션 추가
    # final_sources가 있고 FAQ가 아니면 RAG 답변으로 간주
        if sources and isinstance(sources, list) and sources != ["FAQ"]:
            # 중복 제거
            seen = set()
            unique_files = []
            for f in sources:
                if f and f.lower() not in seen:
                    unique_files.append(f)
                    seen.add(f.lower())
            
            if unique_files:
                # 참고 문서를 Markdown 형식으로 추가
                ref_section = "\n\n---\n\n**참고 문서:**\n\n"
                ref_items = [f"- {f}" for f in unique_files]
                ref_section += "\n".join(ref_items)
                answer = answer.rstrip() + ref_section
            
            # 연관 검색어 추가 (relevance_score >= 0.7인 문서들에서 추출)
            related_queries = state.get("related_queries")
            if related_queries and isinstance(related_queries, list) and len(related_queries) > 0:
                related_section = "\n\n---\n\n**연관 검색어:**\n\n"
                related_items = [f"- {q}" for q in related_queries]
                related_section += "\n".join(related_items)
                answer = answer.rstrip() + related_section
    
    # 필수 필드 기본값 보장 (None 방지) - ChatResponse 모델 검증 통과를 위해 필수
    # 🚨 필수 필드에 대한 기본값 보장 (None 방지) 🚨
    
    # token_usage: dict (필수)
    token_usage = state.get("token_usage")
    if token_usage is None or not isinstance(token_usage, dict):
        token_usage = {}
    
    # route: str (필수)
    route = state.get("route")
    if route is None or not isinstance(route, str) or not route.strip():
        route = "unknown"
    
    # session_id: str (필수) - messages.py에서 자동 생성하므로 여기서는 빈 문자열 유지
    # ⚠️ 세션 만료 시에도 session_id는 유지되어야 함 (요청에서 받은 session_id 사용)
    session_id = state.get("session_id")
    if session_id is None:
        session_id = ""  # None이면 빈 문자열 (messages.py에서 자동 생성)
    elif not isinstance(session_id, str):
        session_id = str(session_id)  # 문자열이 아니면 변환
    # 빈 문자열이어도 messages.py에서 처리하므로 그대로 전달
    
    # faq_count: int (필수)
    faq_count = state.get("faq_count")
    if faq_count is None or not isinstance(faq_count, int):
        faq_count = 0
    
    # cache_hit: bool (필수)
    cache_hit = state.get("cache_hit")
    if cache_hit is None or not isinstance(cache_hit, bool):
        cache_hit = False
    
    # blocked: bool (필수)
    blocked = state.get("blocked")
    if blocked is None or not isinstance(blocked, bool):
        blocked = False
    
    # answer가 None이면 빈 문자열로 처리 (ChatResponse 모델 검증 통과)
    if answer is None:
        answer = ""
    elif not isinstance(answer, str):
        answer = str(answer)
    
    # related_queries 가져오기
    related_queries = state.get("related_queries")
    if related_queries is None or not isinstance(related_queries, list):
        related_queries = []
    
    return {
        "answer": answer,  # str 타입 보장 (None이면 빈 문자열)
        "sources": sources,
        "blocked": blocked,  # bool 타입 보장
        "block_reason": state.get("block_reason"),
        "token_usage": token_usage,  # dict 타입 보장
        "route": route,  # str 타입 보장
        "faq_count": faq_count,  # int 타입 보장
        "cache_hit": cache_hit,  # bool 타입 보장
        "session_id": session_id,  # str 타입 보장
        "intent_type": state.get("intent_type"),
        "intent_category": state.get("intent_category"),
        "model_used": state.get("model_used"),
        "related_queries": related_queries,  # 연관 검색어
        # format_response에서도 final_answer와 final_sources를 반환하여 chat_usecase에서 사용 가능하도록
        "final_answer": answer,
        "final_sources": sources
    }

def prepare_rerun_state(
    previous_state: dict,
    new_message: Optional[str] = None
) -> dict:
    """재질문을 위한 상태 준비 (같은 질문을 다시 요청)
    
    Args:
        previous_state: 이전 질문의 상태 (전체 state)
        new_message: 새로운 메시지 (None이면 같은 질문 재요청)
    
    Returns:
        재질문을 위한 새로운 상태
    """
    # 재질문: 같은 질문을 다시 요청 (new_message가 None이면 이전 질문 사용)
    user_message = new_message if new_message else previous_state.get("user_message", "")
    
    # 이전 대화 히스토리 유지 (재질문이므로 이전 답변은 제외하고 질문만 다시)
    chat_history = previous_state.get("chat_history", [])
    if not isinstance(chat_history, list):
        chat_history = []
    
    # 재질문이므로 이전 답변은 히스토리에 포함하지 않음 (같은 질문을 다시 요청)
    # 단, 이전 대화 맥락은 유지
    
    # 새로운 초기 상태 생성 (같은 질문으로)
    new_state = create_initial_state(
        user_message=user_message,
        user_id=previous_state["user_id"],
        session_id=previous_state["session_id"],
        chat_history=chat_history,  # 이전 대화 맥락 유지
        top_p=previous_state.get("top_p", 0.8),
        chatbot_settings=previous_state.get("chatbot_settings")
    )
    
    new_state["needs_rerun"] = True
    new_state["previous_questions"] = previous_state.get("previous_questions", [])
    
    return new_state

