"""챗봇 설정 서비스 (Domain Layer)"""
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum

from app.domain.ports.repository_port import ConfigRepositoryPort
from app.infrastructure.persistence.models.chatbot_config_model import (
    PersonaType,
    ResponseLength,
    LLMModel
)


class PersonaTypeEnum(str, Enum):
    """페르소나 타입"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"


class ResponseLengthEnum(str, Enum):
    """응답 길이"""
    SHORT = "short"
    NORMAL = "normal"
    LONG = "long"
    VERY_LONG = "very_long"


@dataclass
class ChatbotSettings:
    """챗봇 설정 데이터 클래스"""
    # Perf: 기본 모델을 claude-haiku-4-5로 변경 (응답 속도 최적화)
    llm_model: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.8
    search_results_count: int = 5
    
    persona_type: str = "professional"
    persona_description: str = "당신은 기업 내부 지식을 관리하는 전문적인 AI 어시스턴트입니다. 정확하고 신뢰할 수 있는 정보를 제공하며, 사용자의 업무 효율을 높이는 것이 목표입니다."
    response_length: str = "normal"
    
    guardrail_keywords: Optional[dict] = None
    thresholds: Optional[dict] = None
    
    def get_guardrail_keywords(self, config_port: Optional['ConfigPort'] = None) -> dict:
        """Guardrail 키워드 반환 (설정 파일에서 로드)
        
        우선순위:
        1. DB 설정 (self.guardrail_keywords)
        2. guardrail_config.json 파일
        3. 빈 딕셔너리 (하드코딩 제거)
        
        Args:
            config_port: 설정 포트 (선택적, 없으면 기본 어댑터 사용)
        
        Returns:
            Guardrail 키워드 딕셔너리
        """
        # DB 설정이 있으면 우선 사용
        if self.guardrail_keywords:
            return self.guardrail_keywords
        
        # guardrail_config.json에서 로드
        try:
            if config_port is None:
                from app.infrastructure.adapters.config.file_config_adapter import FileConfigAdapter
                config_port = FileConfigAdapter()
            
            guardrail_config = config_port.load_file("guardrail_config.json")
            if guardrail_config and isinstance(guardrail_config, dict):
                keywords = guardrail_config.get("guardrail_keywords")
                if keywords and isinstance(keywords, dict):
                    return keywords
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"guardrail_config.json에서 키워드를 로드할 수 없습니다: {e}. 빈 키워드 사용.")
        
        return {
            "blocked_topic_keywords": [],
            "business_keywords": [],
            "casual_keywords": [],
            "bot_initiated_casual_keywords": []
        }
    
    def get_thresholds(self, config_port: Optional['ConfigPort'] = None) -> dict:
        """임계값 반환 (설정 파일에서 로드)
        
        우선순위:
        1. DB 설정 (self.thresholds)
        2. guardrail_config.json 파일
        3. 최소한의 기본값 (하드코딩 최소화)
        
        Args:
            config_port: 설정 포트 (선택적, 없으면 기본 어댑터 사용)
        
        Returns:
            임계값 딕셔너리
        """
        if self.thresholds:
            return self.thresholds
        
        try:
            if config_port is None:
                from app.infrastructure.adapters.config.file_config_adapter import FileConfigAdapter
                config_port = FileConfigAdapter()
            
            guardrail_config = config_port.load_file("guardrail_config.json")
            if guardrail_config and isinstance(guardrail_config, dict):
                thresholds = guardrail_config.get("thresholds")
                if thresholds and isinstance(thresholds, dict):
                    return thresholds
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"guardrail_config.json에서 임계값을 로드할 수 없습니다: {e}. 기본값 사용.")
        
        return {
            "faq_search_threshold": 0.60,
            "faq_min_confidence": 0.75,
            "faq_generation_min_confidence": 0.85,
            "rag_confidence_threshold": 0.7
        }
    
    @classmethod
    def from_config_dict(cls, config_dict: Optional[dict]) -> "ChatbotSettings":
        """설정 dict에서 ChatbotSettings 생성
        
        Args:
            config_dict: 설정 dict (ConfigRepositoryPort에서 반환된 형태)
        
        Returns:
            ChatbotSettings: 챗봇 설정
        """
        if config_dict:
            # enum을 실제 모델 ID로 변환
            llm_model_str = config_dict.get('llm_model', '')
            try:
                # LLMModel enum에서 모델 ID 가져오기
                from app.infrastructure.persistence.models.chatbot_config_model import LLMModel
                if isinstance(llm_model_str, str) and hasattr(LLMModel, llm_model_str):
                    llm_model_enum = getattr(LLMModel, llm_model_str)
                    llm_model_id = LLMModel.get_model_id(llm_model_enum)
                else:
                    llm_model_id = llm_model_str  # 이미 모델 ID인 경우
            except Exception:
                llm_model_id = llm_model_str  # 변환 실패 시 원본 사용
            
            return cls(
                llm_model=llm_model_id,
                temperature=config_dict.get('temperature', 0.7),
                max_tokens=config_dict.get('max_tokens', 2000),
                top_p=config_dict.get('top_p', 0.8),
                search_results_count=config_dict.get('search_results_count', 5),
                persona_type=config_dict.get('persona_type', 'professional'),
                persona_description=config_dict.get('persona_description', ''),
                response_length=config_dict.get('response_length', 'normal'),
                guardrail_keywords=config_dict.get('guardrail_keywords'),
                thresholds=config_dict.get('thresholds'),
            )
        return cls()  # 기본값 반환
    
    @classmethod
    def from_config(cls, config: Optional[Any]) -> "ChatbotSettings":
        """ChatbotConfig 엔티티에서 설정 생성 (하위 호환성)
        
        Args:
            config: ChatbotConfig 엔티티 또는 dict
        
        Returns:
            ChatbotSettings: 챗봇 설정
        """
        if config is None:
            return cls()
        
        # dict인 경우
        if isinstance(config, dict):
            return cls.from_config_dict(config)
        
        # ChatbotConfig 엔티티인 경우
        try:
            from app.infrastructure.persistence.models.chatbot_config_model import LLMModel
            llm_model_id = LLMModel.get_model_id(config.llm_model)
            
            return cls(
                llm_model=llm_model_id,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                search_results_count=config.search_results_count,
                persona_type=config.persona_type.value,
                persona_description=config.persona_description,
                response_length=config.response_length.value,
                guardrail_keywords=config.guardrail_keywords,
                thresholds=config.thresholds,
            )
        except Exception:
            return cls()  # 변환 실패 시 기본값 반환


def get_chatbot_settings(
    company_id: str,
    config_repository: Optional[ConfigRepositoryPort] = None
) -> ChatbotSettings:
    """회사별 챗봇 설정 조회
    
    Args:
        company_id: 회사 ID
        config_repository: Config Repository 포트 (None이면 하위 호환성으로 직접 생성)
        
    Returns:
        ChatbotSettings: 챗봇 설정 (없으면 기본값)
    """
    if config_repository:
        # 포트를 통해 조회
        config_dict = config_repository.find_by_company_id(company_id)
        
        # 없으면 기본 설정 조회
        if not config_dict:
            config_dict = config_repository.find_default()
        
        return ChatbotSettings.from_config_dict(config_dict)
    else:
        # 하위 호환성: 직접 생성
        from app.infrastructure.persistence.session import db_session
        from app.infrastructure.persistence.repositories.config_repository import ConfigRepository
        
        with db_session() as db:
            repo = ConfigRepository(db)
            config = repo.find_by_company_id(company_id)
            if not config:
                config = repo.find_default()
            return ChatbotSettings.from_config(config)

