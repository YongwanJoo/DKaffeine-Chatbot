from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.repositories.chat_model_config_repository import ChatModelConfigRepository

class ChatbotConfigSyncService:
    # 클래스 레벨 캐시 (모든 인스턴스 공유)
    _cache: Dict[int, Dict[str, Any]] = {}

    def __init__(self, session: AsyncSession):
        self.repository = ChatModelConfigRepository(session)

    async def get_config_for_model(self, chat_model_id: int) -> Optional[Dict[str, Any]]:
        """
        특정 모델의 최신 설정을 조회하여 딕셔너리 형태로 반환합니다.
        API를 통해 캐시가 무효화되기 전까지는 캐시된 값을 반환합니다.
        """
        # 캐시 확인
        if chat_model_id in self._cache:
            return self._cache[chat_model_id]

        # DB에서 조회 (chat_model과 조인하여 모델 정보도 함께 가져오기)
        result = await self.repository.get_latest_config_with_model(chat_model_id)
        
        if not result:
            return None
        
        config, chat_model = result
        
        model_name = chat_model.chat_model_name if chat_model else None
        llm_model_id = self._map_model_name_to_id(model_name)
            
        config_dict = {
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "max_tokens": config.max_tokens,
            "search_results_count": int(config.search_result_count),  # rag_service.py에서 사용하는 키 이름으로 통일
            "persona_type": config.persona_type,
            "persona_description": config.system_prompt if config.system_prompt else "",  # rag_service.py에서 persona_description을 찾음
            "system_prompt": config.system_prompt,  # 하위 호환성을 위해 유지
            "response_length": config.response_length if config.response_length else "normal",
            "llm_model": llm_model_id  # chat_model에서 가져온 모델 ID
        }
        
        # 캐시 업데이트
        self._cache[chat_model_id] = config_dict
        
        return config_dict

    @staticmethod
    def _map_model_name_to_id(model_name: Optional[str]) -> str:
        """
        chat_model_name을 실제 Bedrock 모델 ID로 변환합니다.
        
        Args:
            model_name: chat_model 테이블의 chat_model_name 값
            
        Returns:
            Bedrock 모델 ID (예: "anthropic.claude-sonnet-4-5-20250929-v1:0")
        """
        if not model_name:
            return "anthropic.claude-sonnet-4-5-20250929-v1:0"  # 기본값
        
        # 이미 모델 ID 형식인지 확인 (예: "anthropic.claude-sonnet-4-5-20250929-v1:0")
        if model_name.startswith("anthropic.") or model_name.startswith("amazon.") or model_name.startswith("cohere."):
            return model_name
        
        # 모델 이름을 모델 ID로 매핑
        # Java 백엔드에서 저장하는 이름 형식에 따라 매핑 로직 추가 가능
        model_name_lower = model_name.lower()
        
        if "sonnet" in model_name_lower and ("4.5" in model_name_lower or "4-5" in model_name_lower):
            return "anthropic.claude-sonnet-4-5-20250929-v1:0"
        elif "sonnet" in model_name_lower and ("3.5" in model_name_lower or "3-5" in model_name_lower):
            return "anthropic.claude-3-5-sonnet-20241022-v2:0"
        elif "opus" in model_name_lower and "3" in model_name_lower:
            return "anthropic.claude-3-opus-20240229-v1:0"
        elif "haiku" in model_name_lower and ("4.5" in model_name_lower or "4-5" in model_name_lower):
            return "anthropic.claude-haiku-4-5-20251001-v1:0"
        
        # 매핑 실패 시 기본값 반환
        return "anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    @classmethod
    def invalidate_cache(cls, chat_model_id: Optional[int] = None):
        """
        캐시를 무효화합니다.
        chat_model_id가 주어지면 해당 모델만, 없으면 전체 캐시를 비웁니다.
        """
        if chat_model_id:
            if chat_model_id in cls._cache:
                del cls._cache[chat_model_id]
        else:
            cls._cache.clear()
