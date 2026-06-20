"""법령 조문 검색 도구."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...chunking.strategies import Chunk
from ...retrieval.embedding_retriever import EmbeddingRetriever

INDEX_DIR = Path(__file__).parent.parent.parent.parent / "data" / "index"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "doc_type": self.chunk.doc_type,
            "source_name": self.chunk.source_name,
            "text": self.chunk.text[:400],
            "score": round(self.score, 4),
            "metadata": self.chunk.metadata,
        }


class StatuteSearchTool:
    """법령 조문 임베딩 검색 도구."""

    name = "search_statutes"
    description = (
        "법령 조문을 검색합니다. 특정 법률 조항, 권리/의무 규정, "
        "절차 규정을 찾을 때 사용합니다. "
        "예: '임대차보증금 반환 의무', '근로계약 해지 절차'"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색할 법령 관련 쿼리 (법률 용어로 구체적으로)",
            },
            "top_k": {
                "type": "integer",
                "description": "반환할 최대 결과 수 (기본 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: EmbeddingRetriever | None = None):
        self._retriever = retriever

    @property
    def retriever(self) -> EmbeddingRetriever:
        if self._retriever is None:
            self._retriever = EmbeddingRetriever()
            index_path = INDEX_DIR / "embedding_index_article.npy"
            if not index_path.exists():
                index_path = INDEX_DIR / "embedding_index.npy"
            self._retriever.load_index(index_path)
        return self._retriever

    def __call__(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raw = self.retriever.search(query, top_k=top_k * 2)
        statute_results = [
            SearchResult(c, s) for c, s in raw if c.doc_type == "statute"
        ]
        return statute_results[:top_k]
