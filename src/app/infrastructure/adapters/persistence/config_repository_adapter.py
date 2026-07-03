"""Config Repository 어댑터"""
from typing import Optional
from sqlalchemy.orm import Session

from app.domain.ports.repository_port import ConfigRepositoryPort
from app.infrastructure.persistence.repositories.config_repository import ConfigRepository
from app.infrastructure.persistence.models.chatbot_config_model import (
    ChatbotConfig,
    PersonaType,
    ResponseLength,
    LLMModel
)


class ConfigRepositoryAdapter(ConfigRepositoryPort):
    """Config Repository 어댑터 구현
    
    ConfigRepositoryPort 인터페이스를 구현하여 설정 데이터 접근을 제공합니다.
    """
    
    def __init__(self, session: Session):
        self.session = session
        self._repo = ConfigRepository(session)
    
    def _config_to_dict(self, config: ChatbotConfig) -> dict:
        """ChatbotConfig 모델을 dict로 변환"""
        return {
            "id": config.id,
            "company_id": config.company_id,
            "llm_model": config.llm_model.value if isinstance(config.llm_model, LLMModel) else str(config.llm_model),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "search_results_count": config.search_results_count,
            "persona_type": config.persona_type.value if isinstance(config.persona_type, PersonaType) else str(config.persona_type),
            "persona_description": config.persona_description,
            "response_length": config.response_length.value if isinstance(config.response_length, ResponseLength) else str(config.response_length),
            "is_active": config.is_active,
            "created_at": config.created_at.isoformat() if config.created_at else None,
            "updated_at": config.updated_at.isoformat() if config.updated_at else None,
        }
    
    def find_by_company_id(self, company_id: str) -> Optional[dict]:
        """회사 ID로 설정 조회"""
        config = self._repo.find_by_company_id(company_id)
        return self._config_to_dict(config) if config else None
    
    def find_default(self) -> Optional[dict]:
        """기본 설정 조회 (company_id='default')"""
        return self.find_by_company_id("default")
    
    def save(self, company_id: str, config: dict) -> dict:
        """설정 저장"""
        # ConfigRepository의 create_or_update 사용
        saved_config = self._repo.create_or_update(
            company_id=company_id,
            llm_model=LLMModel[config.get('llm_model', 'CLAUDE_SONNET_4_5')] if isinstance(config.get('llm_model'), str) else config.get('llm_model', LLMModel.CLAUDE_SONNET_4_5),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 2000),
            top_p=config.get('top_p', 0.9),
            search_results_count=config.get('search_results_count', 5),
            persona_type=PersonaType[config.get('persona_type', 'PROFESSIONAL')] if isinstance(config.get('persona_type'), str) else config.get('persona_type', PersonaType.PROFESSIONAL),
            persona_description=config.get('persona_description', ''),
            response_length=ResponseLength[config.get('response_length', 'NORMAL')] if isinstance(config.get('response_length'), str) else config.get('response_length', ResponseLength.NORMAL),
        )
        return self._config_to_dict(saved_config)

