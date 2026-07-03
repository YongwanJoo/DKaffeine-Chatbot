"""챗봇 설정 Repository"""
import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.infrastructure.persistence.models.chatbot_config_model import ChatbotConfig, PersonaType, ResponseLength, LLMModel

logger = logging.getLogger(__name__)


class ConfigRepository:
    """챗봇 설정 데이터베이스 접근 레이어"""

    def __init__(self, session: Session):
        self.session = session

    def find_by_company_id(self, company_id: str) -> Optional[ChatbotConfig]:
        """회사 ID로 설정 조회 (순수 읽기)
        
        읽기 전용 메서드입니다. 데이터 마이그레이션은 수행하지 않습니다.
        구버전 enum 값이 감지되면 로그만 남기고 None을 반환합니다.
        마이그레이션이 필요한 경우 migrate_legacy_enum_values()를 별도로 호출하세요.
        
        Args:
            company_id: 회사 ID
        
        Returns:
            ChatbotConfig 인스턴스 또는 None (구버전 enum 값이 있으면 None)
        """
        try:
            # 먼저 raw SQL로 llm_model 값을 확인 (enum 변환 전, 텍스트로 읽기)
            # 읽기만 수행, UPDATE는 하지 않음
            result = self.session.execute(
                text("""
                    SELECT llm_model::text as llm_model_text
                    FROM chatbot_config 
                    WHERE company_id = :company_id AND is_active = true
                """),
                {"company_id": company_id}
            ).first()
            
            # 구버전 enum 값이 있는지 확인 (읽기만 수행, 마이그레이션은 하지 않음)
            if result:
                llm_model_value = result[0]
                # 구버전 GPT 모델 값들 또는 긴 문자열 모델 ID
                old_models = ['GPT_4O', 'GPT_4O_MINI', 'GPT_4', 'GPT_4_TURBO']
                # 긴 문자열 모델 ID도 감지 (PostgreSQL enum에 저장 불가능한 값)
                is_long_string = llm_model_value and len(str(llm_model_value)) > 50
                
                if str(llm_model_value) in old_models or is_long_string:
                    logger.warning(
                        f"구버전 llm_model 값 감지: company_id={company_id}, "
                        f"value={llm_model_value}. 마이그레이션이 필요하지만 읽기 요청이므로 수행하지 않음. "
                        f"별도로 migrate_legacy_enum_values()를 호출하거나 create_or_update()를 사용하세요."
                    )
                    # 읽기 전용 복제본이나 트랜잭션 격리 문제를 피하기 위해 None 반환
                    # 기본값을 사용하도록 함
                    return None
            
            # 정상적인 경우 ORM으로 조회
            config = self.session.query(ChatbotConfig).filter(
                ChatbotConfig.company_id == company_id,
                ChatbotConfig.is_active == True
            ).first()
            
            return config
        except Exception as e:
            # PostgreSQL 연결 실패는 예상 가능한 상황이므로 WARNING 레벨로 처리
            # (PostgreSQL이 선택적이거나 일시적으로 사용 불가능한 경우)
            error_msg = str(e)
            if "connection failed" in error_msg.lower() or "operationalerror" in error_msg.lower():
                logger.warning(
                    f"PostgreSQL 연결 실패로 설정 조회 불가: company_id={company_id}. "
                    f"기본값을 사용합니다. (연결 정보 확인: .secrets.toml의 [postgres] 섹션)"
                )
            else:
                logger.error(f"설정 조회 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 rollback 후 None 반환 (기본값 사용)
            try:
                self.session.rollback()
            except Exception:
                pass
            return None
    
    def migrate_legacy_enum_values(self, company_id: str) -> bool:
        """구버전 enum 값을 마이그레이션 (별도 쓰기 작업)
        
        읽기와 쓰기를 분리하기 위해 마이그레이션을 별도 메서드로 분리했습니다.
        이 메서드는 쓰기 작업이므로 읽기 전용 복제본에서는 사용할 수 없습니다.
        
        Args:
            company_id: 회사 ID
        
        Returns:
            마이그레이션 성공 여부
        """
        try:
            # 먼저 raw SQL로 llm_model 값을 확인 (enum 변환 전, 텍스트로 읽기)
            result = self.session.execute(
                text("""
                    SELECT llm_model::text as llm_model_text
                    FROM chatbot_config 
                    WHERE company_id = :company_id AND is_active = true
                """),
                {"company_id": company_id}
            ).first()
            
            # 구버전 enum 값이 있는지 확인하고 마이그레이션
            if result:
                llm_model_value = result[0]
                # 구버전 GPT 모델 값들 또는 긴 문자열 모델 ID
                old_models = ['GPT_4O', 'GPT_4O_MINI', 'GPT_4', 'GPT_4_TURBO']
                # 긴 문자열 모델 ID도 감지 (PostgreSQL enum에 저장 불가능한 값)
                is_long_string = llm_model_value and len(str(llm_model_value)) > 50
                
                if str(llm_model_value) in old_models or is_long_string:
                    logger.info(
                        f"구버전 llm_model 값 마이그레이션 시작: company_id={company_id}, "
                        f"value={llm_model_value}"
                    )
                    # Raw SQL로 직접 업데이트 (짧은 enum 이름 사용)
                    # CAST를 사용하여 enum 타입으로 변환
                    self.session.execute(
                        text("""
                            UPDATE chatbot_config 
                            SET llm_model = :new_model::llmmodel
                            WHERE company_id = :company_id AND is_active = true
                        """),
                        {
                            "new_model": LLMModel.CLAUDE_SONNET_4_5.value,  # "CLAUDE_SONNET_4_5"
                            "company_id": company_id
                        }
                    )
                    self.session.commit()
                    logger.info(f"구버전 llm_model 값 마이그레이션 완료: company_id={company_id}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"마이그레이션 중 오류 발생: {e}", exc_info=True)
            try:
                self.session.rollback()
            except Exception:
                pass
            return False

    def find_default(self) -> Optional[ChatbotConfig]:
        """기본 설정 조회 (company_id='default')"""
        return self.find_by_company_id("default")

    def create_or_update(
        self,
        company_id: str,
        llm_model: LLMModel = LLMModel.CLAUDE_SONNET_4_5,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        search_results_count: int = 5,
        persona_type: PersonaType = PersonaType.PROFESSIONAL,
        persona_description: str = "",
        response_length: ResponseLength = ResponseLength.NORMAL,
    ) -> ChatbotConfig:
        """설정 생성 또는 업데이트
        
        쓰기 작업이므로 마이그레이션이 필요한 경우 자동으로 수행합니다.
        """
        config = self.find_by_company_id(company_id)
        
        # 읽기에서 구버전 enum 값이 감지되어 None이 반환된 경우 마이그레이션 시도
        if config is None:
            # 마이그레이션이 필요한 경우 시도 (쓰기 작업이므로 가능)
            migrated = self.migrate_legacy_enum_values(company_id)
            if migrated:
                # 마이그레이션 후 다시 조회
                config = self.find_by_company_id(company_id)
        
        if config:
            # 업데이트
            config.llm_model = llm_model
            config.temperature = temperature
            config.max_tokens = max_tokens
            config.top_p = top_p
            config.search_results_count = search_results_count
            config.persona_type = persona_type
            config.persona_description = persona_description
            config.response_length = response_length
        else:
            # 생성
            config = ChatbotConfig(
                company_id=company_id,
                llm_model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                search_results_count=search_results_count,
                persona_type=persona_type,
                persona_description=persona_description,
                response_length=response_length,
            )
            self.session.add(config)
        
        self.session.commit()
        self.session.refresh(config)
        return config

