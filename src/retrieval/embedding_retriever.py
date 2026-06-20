"""임베딩 기반 벡터 검색 (sentence-transformers + cosine similarity).

1주차: 단순 in-memory numpy 인덱스 (외부 벡터DB 없음).
2주차 실험: 다른 모델, ONNX 최적화 등으로 교체.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..chunking.strategies import Chunk

DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 64


class EmbeddingRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"임베딩 모델 로드 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None  # shape: (N, dim)

    def build_index(self, chunks: list[Chunk], cache_path: Path | None = None) -> None:
        """청크 목록으로 임베딩 인덱스 구축."""
        if cache_path and cache_path.exists():
            print(f"캐시에서 인덱스 로드: {cache_path}")
            data = np.load(str(cache_path), allow_pickle=True).item()
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]
        print(f"{len(texts)}개 청크 임베딩 계산 중...")
        embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="임베딩"):
            batch = texts[i : i + BATCH_SIZE]
            vecs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embeddings.append(vecs)
        self.embeddings = np.vstack(embeddings).astype(np.float32)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), {"chunks": self.chunks, "embeddings": self.embeddings})
            print(f"인덱스 캐시 저장: {cache_path}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """쿼리와 코사인 유사도 기준 상위 top_k 청크 반환."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q_vec = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = self.embeddings @ q_vec  # cosine similarity (normalized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def search_by_doc_type(
        self, query: str, doc_type: str, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """특정 doc_type (statute/precedent)만 대상으로 검색."""
        results = self.search(query, top_k=top_k * 3)
        filtered = [(c, s) for c, s in results if c.doc_type == doc_type]
        return filtered[:top_k]

    def save_index(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), {"chunks": self.chunks, "embeddings": self.embeddings})

    def load_index(self, path: Path) -> None:
        data = np.load(str(path), allow_pickle=True).item()
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
