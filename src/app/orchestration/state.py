"""
DEPRECATED: 이 파일은 더 이상 사용되지 않습니다.

ChatState와 merge_token_usage는 다음 위치로 이동되었습니다:
- app.domain.entities.chat_state.ChatState
- app.domain.entities.chat_state.merge_token_usage

이 파일은 하위 호환성을 위해 domain 버전을 re-export합니다.
새 코드에서는 app.domain.entities.chat_state를 직접 사용하세요.
"""

# Domain 버전을 re-export (Single Source of Truth)
from app.domain.entities.chat_state import (
    ChatState,
    merge_token_usage,
    ensure_chat_state,
    to_state_dict,
)

__all__ = ["ChatState", "merge_token_usage", "ensure_chat_state", "to_state_dict"]

