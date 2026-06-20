"""판례 검색 도구."""

from __future__ import annotations

from pathlib import Path

from ...retrieval.embedding_retriever import EmbeddingRetriever
from .statute_search import SearchResult

INDEX_DIR = Path(__file__).parent.parent.parent.parent / "data" / "index"


class PrecedentSearchTool:
    """판례 임베딩 검색 도구."""

    name = "search_precedents"
    description = (
        "법원 판례를 검색합니다. 실제 사례에서 법원이 어떻게 판단했는지, "
        "유사한 분쟁의 결과를 찾을 때 사용합니다. "
        "예: '전세금 반환 거부 판례', '부당해고 구제 판례'"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색할 판례 관련 쿼리 (분쟁 상황, 법률 쟁점 위주로)",
            },
            "top_k": {
                "type": "integer",
                "description": "반환할 최대 결과 수 (기본 3)",
                "default": 3,
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

    def __call__(self, query: str, top_k: int = 3) -> list[SearchResult]:
        raw = self.retriever.search(query, top_k=top_k * 3)
        prec_results = [
            SearchResult(c, s) for c, s in raw if c.doc_type == "precedent"
        ]
        return prec_results[:top_k]


class CombinedSearchTool:
    """법령 + 판례 통합 검색 도구."""

    name = "search_combined"
    description = (
        "법령 조문과 판례를 함께 검색합니다. "
        "일반인의 상담형 질문(어떻게 해야 하나요? 권리가 있나요?)처럼 "
        "법령 근거와 실제 사례를 모두 필요로 할 때 사용합니다."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색 쿼리 (구어체 질문 그대로 사용 가능)",
            },
            "statute_k": {
                "type": "integer",
                "description": "법령 검색 결과 수 (기본 3)",
                "default": 3,
            },
            "precedent_k": {
                "type": "integer",
                "description": "판례 검색 결과 수 (기본 2)",
                "default": 2,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: EmbeddingRetriever | None = None):
        from .statute_search import StatuteSearchTool
        self._statute_tool  = StatuteSearchTool(retriever)
        self._prec_tool     = PrecedentSearchTool(retriever)

    def __call__(
        self, query: str, statute_k: int = 3, precedent_k: int = 2
    ) -> dict[str, list[SearchResult]]:
        return {
            "statutes":   self._statute_tool(query, top_k=statute_k),
            "precedents": self._prec_tool(query, top_k=precedent_k),
        }
