from __future__ import annotations

from typing import List, Tuple, Protocol

from langchain_core.documents import Document


class BaseRetriever(Protocol):
    """문서 검색기 인터페이스.

    Bedrock Knowledge Base, OpenSearch, FAISS 등으로 교체 가능하도록 추상화.
    """

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        ...


__all__ = ["BaseRetriever"]


