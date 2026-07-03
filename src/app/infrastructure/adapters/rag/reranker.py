"""Reranker 모듈 (Deprecated - 하위 호환성 유지)

이 파일은 하위 호환성을 위해 유지됩니다.
새로운 코드에서는 다음과 같이 import하세요:

    from app.infrastructure.adapters.rag.rerankers import CrossEncoderReranker, BedrockReranker

기존 코드 호환성:

    from app.infrastructure.adapters.rag.reranker import CrossEncoderReranker, BedrockReranker
"""
import warnings

# 하위 호환성을 위한 re-export
from .rerankers import CrossEncoderReranker, BedrockReranker

# Deprecation warning
warnings.warn(
    "Importing from 'rag.reranker' is deprecated. "
    "Please use 'from app.infrastructure.adapters.rag.rerankers import CrossEncoderReranker, BedrockReranker' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["CrossEncoderReranker", "BedrockReranker"]
