"""임베딩 추론 캐싱.

동일 쿼리에 대해 임베딩 벡터를 재계산하지 않고 캐시에서 반환.
실시간 서빙에서 반복 질문(FAQ성 질문)의 latency를 크게 줄일 수 있다.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from functools import wraps
from typing import Callable

import numpy as np


class LRUEmbeddingCache:
    """LRU(Least Recently Used) 방식 임베딩 벡터 캐시."""

    def __init__(self, max_size: int = 512):
        self.max_size  = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits   = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        k = self._key(text)
        if k in self._cache:
            self._cache.move_to_end(k)  # LRU 갱신
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, text: str, vec: np.ndarray) -> None:
        k = self._key(text)
        self._cache[k] = vec
        self._cache.move_to_end(k)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # 가장 오래된 항목 제거

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "size":     len(self._cache),
            "max_size": self.max_size,
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = self.misses = 0


def cached_encode(cache: LRUEmbeddingCache, encode_fn: Callable) -> Callable:
    """encode 함수에 LRU 캐시를 씌우는 래퍼."""
    @wraps(encode_fn)
    def wrapper(texts: list[str], **kwargs) -> np.ndarray:
        results: list[np.ndarray | None] = [cache.get(t) for t in texts]
        miss_indices = [i for i, r in enumerate(results) if r is None]

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            new_vecs   = encode_fn(miss_texts, **kwargs)
            for idx, vec in zip(miss_indices, new_vecs):
                cache.put(texts[idx], vec)
                results[idx] = vec

        return np.stack(results)  # type: ignore[arg-type]
    return wrapper
