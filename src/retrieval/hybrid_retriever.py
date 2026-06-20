"""하이브리드 검색 — BM25 + 임베딩 (Reciprocal Rank Fusion).

RRF(Reciprocal Rank Fusion) 방식:
  각 검색기의 순위(rank)를 기반으로 점수를 합산.
  score(doc) = Σ 1 / (k + rank_i(doc))   (k=60 기본값)

장점: 두 검색기의 점수 스케일이 달라도 순위 기반이라 자연스럽게 합산됨.

사용법:
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(embedding_retriever, bm25_retriever)
    results = retriever.search("전세금을 못 받았어요", top_k=5)
"""

from __future__ import annotations

from ..chunking.strategies import Chunk
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever

DEFAULT_RRF_K = 60


class HybridRetriever:
    """임베딩 + BM25 Reciprocal Rank Fusion 하이브리드 검색기."""

    def __init__(
        self,
        embedding: EmbeddingRetriever,
        bm25: BM25Retriever,
        rrf_k: int = DEFAULT_RRF_K,
        emb_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.embedding = embedding
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.emb_weight = emb_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """RRF 기반 하이브리드 검색. (chunk, score) 리스트 반환."""
        candidate_k = top_k * 4  # 충분히 많이 뽑아서 합산

        emb_results = self.embedding.search(query, top_k=candidate_k)
        bm25_results = self.bm25.search(query, top_k=candidate_k)

        # chunk_id → Chunk 매핑
        chunk_map: dict[str, Chunk] = {}
        for c, _ in emb_results + bm25_results:
            chunk_map[c.chunk_id] = c

        # 순위 추출
        emb_rank: dict[str, int] = {c.chunk_id: i for i, (c, _) in enumerate(emb_results)}
        bm25_rank: dict[str, int] = {c.chunk_id: i for i, (c, _) in enumerate(bm25_results)}

        # RRF 점수 계산
        all_ids = set(emb_rank) | set(bm25_rank)
        scored: list[tuple[Chunk, float]] = []
        for cid in all_ids:
            emb_s = self.emb_weight / (self.rrf_k + emb_rank.get(cid, candidate_k))
            bm25_s = self.bm25_weight / (self.rrf_k + bm25_rank.get(cid, candidate_k))
            scored.append((chunk_map[cid], emb_s + bm25_s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_doc_type(
        self, query: str, doc_type: str, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        results = self.search(query, top_k=top_k * 3)
        filtered = [(c, s) for c, s in results if c.doc_type == doc_type]
        return filtered[:top_k]
