"""BM25 키워드 검색 (rank-bm25 + kiwipiepy 형태소 분석).

1주차에 구현하되 실제 비교 실험은 2주차에 수행.
"""

from __future__ import annotations

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from ..chunking.strategies import Chunk

_kiwi = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize_ko(text: str) -> list[str]:
    """한국어 형태소 분석 후 명사/동사/형용사 추출."""
    kiwi = _get_kiwi()
    tokens = kiwi.tokenize(text)
    # 명사(NN*), 동사(VV*), 형용사(VA*), 어근(XR) 추출
    keep_pos = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "XR", "SL"}
    return [t.form for t in tokens if t.tag in keep_pos]


class BM25Retriever:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def build_index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        print(f"BM25 인덱스 구축 중: {len(chunks)}개 청크")
        tokenized = [tokenize_ko(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.bm25 is None or not self.chunks:
            return []
        q_tokens = tokenize_ko(query)
        scores = self.bm25.get_scores(q_tokens)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
