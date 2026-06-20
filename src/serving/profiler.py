"""임베딩 추론 성능 프로파일링.

측정 항목:
  - 단일 쿼리 인코딩 latency (캐시 미스 / 히트)
  - 배치 처리 처리량 (queries/sec)
  - cProfile을 이용한 함수별 시간 분석

사용법:
    python scripts/profile_serving.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import statistics
import time
from pathlib import Path

import numpy as np


def measure_latency(fn, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> dict:
    """함수 실행 시간을 n_runs회 측정하여 통계 반환."""
    # warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms":   round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "min_ms":    round(min(times), 2),
        "max_ms":    round(max(times), 2),
        "p95_ms":    round(sorted(times)[int(len(times) * 0.95)], 2),
        "n_runs":    n_runs,
    }


def profile_function(fn, *args, **kwargs) -> str:
    """cProfile로 함수 실행을 프로파일링하고 상위 20개 함수 출력."""
    pr = cProfile.Profile()
    pr.enable()
    fn(*args, **kwargs)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    return buf.getvalue()


def run_batch_throughput(
    encode_fn, queries: list[str], batch_sizes: list[int]
) -> list[dict]:
    """다양한 배치 크기에서 처리량(queries/sec) 측정."""
    results = []
    for bs in batch_sizes:
        batches = [queries[i : i + bs] for i in range(0, len(queries), bs)]
        t0 = time.perf_counter()
        for batch in batches:
            encode_fn(batch)
        elapsed = time.perf_counter() - t0
        qps = len(queries) / elapsed
        results.append({
            "batch_size": bs,
            "elapsed_s":  round(elapsed, 3),
            "qps":        round(qps, 1),
        })
    return results
