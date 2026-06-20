"""임베딩 서빙 최적화 실험 스크립트.

측정 항목:
  1. 단일 쿼리 latency (캐시 없음 vs 캐시 히트)
  2. 배치 크기별 처리량 (1, 8, 32, 64)
  3. cProfile 함수 분석

최적화 적용:
  - LRU 임베딩 캐시 (동일 쿼리 재사용)
  - 배치 인코딩 (단일 쿼리 반복보다 배치가 유리한 구간 확인)

실행:
    python scripts/profile_serving.py

산출물:
    reports/serving_profile_YYYYMMDD_HHMM.md
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 프로파일링용 샘플 질문
SAMPLE_QUERIES = [
    "전세금을 못 받았어요. 어떻게 해야 하나요?",
    "임대차보증금반환청구권 행사 절차",
    "회사에서 갑자기 해고당했어요. 부당해고 구제 방법이 있나요?",
    "근로기준법 해고 예고 절차",
    "인터넷 쇼핑몰에서 물건을 샀는데 환불을 거부해요.",
    "전자상거래 청약철회 기간",
    "임대인이 계약 갱신을 거부해요. 계속 살 수 있나요?",
    "주택임대차보호법 계약갱신요구권",
    "월세를 3개월 이상 못 냈는데 집주인이 나가라고 해요.",
    "임대차 해지 통보 기간",
]


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from src.serving.profiler import measure_latency, profile_function, run_batch_throughput
    from src.serving.cache import LRUEmbeddingCache, cached_encode

    print("임베딩 모델 로드 중...")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    encode_raw = lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    print("\n[1] 단일 쿼리 latency 측정 (캐시 없음)")
    no_cache_stats = measure_latency(
        encode_raw, [SAMPLE_QUERIES[0]], n_runs=20, warmup=3
    )
    print(f"  평균: {no_cache_stats['mean_ms']:.1f}ms  P95: {no_cache_stats['p95_ms']:.1f}ms")

    print("\n[2] LRU 캐시 적용 후 latency")
    cache = LRUEmbeddingCache(max_size=256)
    cached_fn = cached_encode(cache, encode_raw)

    # 첫 호출 = 캐시 미스
    cache_miss_stats = measure_latency(
        lambda: cached_fn([SAMPLE_QUERIES[0]]), n_runs=5, warmup=0
    )
    print(f"  캐시 미스: {cache_miss_stats['mean_ms']:.1f}ms")

    # 사전에 캐시 채우기
    for q in SAMPLE_QUERIES:
        cached_fn([q])

    # 두 번째 호출 = 캐시 히트
    cache_hit_stats = measure_latency(
        lambda: cached_fn([SAMPLE_QUERIES[0]]), n_runs=20, warmup=2
    )
    print(f"  캐시 히트: {cache_hit_stats['mean_ms']:.2f}ms  ({cache.stats()})")

    speedup = no_cache_stats["mean_ms"] / max(cache_hit_stats["mean_ms"], 0.01)
    print(f"  속도 향상: {speedup:.0f}x")

    print("\n[3] 배치 크기별 처리량 (100개 쿼리 기준)")
    all_queries = (SAMPLE_QUERIES * 10)[:100]
    batch_results = run_batch_throughput(encode_raw, all_queries, batch_sizes=[1, 8, 32, 64])
    for r in batch_results:
        print(f"  batch={r['batch_size']:3d}: {r['qps']:.1f} queries/sec")

    print("\n[4] cProfile 분석 (단일 쿼리)")
    profile_output = profile_function(encode_raw, [SAMPLE_QUERIES[0]])

    # 리포트 생성
    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    best_batch = max(batch_results, key=lambda r: r["qps"])

    md = f"""# 서빙 최적화 실험 결과

실험일: {date_str}
모델: jhgan/ko-sroberta-multitask (384차원)
환경: CPU only (GPU 없음)

---

## 1. 단일 쿼리 Latency

| 조건 | 평균 | P95 |
|------|------|-----|
| 캐시 없음 (매번 인코딩) | {no_cache_stats['mean_ms']:.1f}ms | {no_cache_stats['p95_ms']:.1f}ms |
| 캐시 미스 (첫 호출) | {cache_miss_stats['mean_ms']:.1f}ms | {cache_miss_stats['p95_ms']:.1f}ms |
| **캐시 히트 (반복 쿼리)** | **{cache_hit_stats['mean_ms']:.2f}ms** | {cache_hit_stats['p95_ms']:.2f}ms |

**속도 향상**: {speedup:.0f}x (캐시 히트 vs 캐시 없음)

- 가설: 동일 쿼리 캐싱으로 latency를 크게 줄일 수 있을 것이다.
- 분석: 캐시 히트 시 {speedup:.0f}배 빠름. FAQ성 반복 질문이 많은 실제 서비스에서 유효한 최적화.
  단, 캐시는 메모리를 사용하므로 max_size 설정이 중요하다 (현재 256개).

---

## 2. 배치 처리 처리량

| 배치 크기 | 처리량 (queries/sec) | 비고 |
|----------|-------------------|------|
{chr(10).join(f"| {r['batch_size']} | {r['qps']:.1f} | {'최고' if r == best_batch else ''}" for r in batch_results)}

- 최적 배치 크기: **{best_batch['batch_size']}** ({best_batch['qps']:.1f} q/s)
- 분석: CPU 환경에서는 배치 크기 증가의 효과가 GPU보다 제한적.
  단일 요청 처리 중심(온라인 서빙)이라면 캐싱이 더 실용적.
  오프라인 대량 처리(인덱스 재빌드)라면 배치 크기 {best_batch['batch_size']} 권장.

---

## 3. 최적화 결론

1. **LRU 임베딩 캐시 도입** (`src/serving/cache.py`):
   - 반복 쿼리에서 {speedup:.0f}x 속도 향상
   - 구현 비용 낮음, 메모리 overhead 최소 (256개 캐시 ≈ 수 MB)
   - 실시간 Streamlit 앱에 즉시 적용 가능

2. **배치 크기 최적화** (인덱스 빌드 시):
   - {best_batch['batch_size']}개 단위 배치가 CPU에서 최고 처리량
   - `scripts/build_index.py`의 BATCH_SIZE를 {best_batch['batch_size']}로 조정 권장

3. **동시 요청 처리**:
   - CPU 단일 프로세스 기준 단일 요청 처리 후 다음 요청 처리
   - 실제 서비스라면 `asyncio` + 임베딩 스레드 분리 또는 ONNX Runtime 변환 검토

---

## 4. cProfile 상세 분석

```
{profile_output[:2000]}
```
"""

    out_path = REPORTS_DIR / f"serving_profile_{stamp}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
