"""간단한 동시 요청 처리 부하 테스트.

동시(concurrent) 요청 시 임베딩 검색 응답 시간을 측정한다.
실제 HTTP 서버 없이 ThreadPoolExecutor로 병렬 호출을 시뮬레이션.

실행:
    python scripts/load_test.py

산출물:
    reports/load_test_YYYYMMDD_HHMM.md
"""

from __future__ import annotations

import concurrent.futures
import datetime
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SAMPLE_QUERIES = [
    "전세금을 못 받았어요. 어떻게 해야 하나요?",
    "회사에서 갑자기 해고당했어요. 부당해고인가요?",
    "인터넷 쇼핑몰 환불 거부 어떻게 해요?",
    "임대인이 계약 갱신을 거부해요.",
    "월세를 3개월 못 냈는데 나가라고 해요.",
    "근로계약서를 안 썼는데 퇴직금 받을 수 있나요?",
    "소비자원에 신고하면 어떻게 되나요?",
    "보증금을 돌려받지 못하고 있어요.",
]


def run_load_test(search_fn, queries: list[str], concurrency: int) -> dict:
    """동시 요청 처리 성능 측정."""
    results_times: list[float] = []
    errors = 0

    def single_request(q: str) -> float:
        t0 = time.perf_counter()
        try:
            search_fn(q)
        except Exception:
            return -1.0
        return (time.perf_counter() - t0) * 1000

    t_total_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request, q) for q in queries]
        for f in concurrent.futures.as_completed(futures):
            elapsed = f.result()
            if elapsed < 0:
                errors += 1
            else:
                results_times.append(elapsed)
    total_elapsed = time.perf_counter() - t_total_start

    return {
        "concurrency":  concurrency,
        "n_requests":   len(queries),
        "errors":       errors,
        "total_s":      round(total_elapsed, 3),
        "throughput":   round(len(queries) / total_elapsed, 2),
        "mean_ms":      round(statistics.mean(results_times), 1) if results_times else 0,
        "p95_ms":       round(sorted(results_times)[int(len(results_times) * 0.95)], 1) if results_times else 0,
        "max_ms":       round(max(results_times), 1) if results_times else 0,
    }


def main() -> None:
    from src.retrieval.embedding_retriever import EmbeddingRetriever

    print("인덱스 로드 중...")
    retriever = EmbeddingRetriever()
    index_path = ROOT / "data" / "index" / "embedding_index_article.npy"
    if not index_path.exists():
        index_path = ROOT / "data" / "index" / "embedding_index.npy"
    retriever.load_index(index_path)

    search_fn = lambda q: retriever.search(q, top_k=5)

    # warmup
    search_fn(SAMPLE_QUERIES[0])

    print("\n동시 요청 처리 측정 (임베딩 검색)")
    print(f"{'동시성':>6} {'처리량(req/s)':>14} {'평균(ms)':>9} {'P95(ms)':>8} {'최대(ms)':>9}")
    print("-" * 52)

    all_results = []
    queries_batch = SAMPLE_QUERIES * 4  # 32개 요청

    for concurrency in [1, 2, 4, 8]:
        r = run_load_test(search_fn, queries_batch, concurrency)
        all_results.append(r)
        print(f"{r['concurrency']:>6}  {r['throughput']:>13}  {r['mean_ms']:>8}  {r['p95_ms']:>7}  {r['max_ms']:>8}")

    # 리포트
    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = "\n".join(
        f"| {r['concurrency']} | {r['throughput']} req/s | {r['mean_ms']}ms | {r['p95_ms']}ms |"
        for r in all_results
    )

    best = max(all_results, key=lambda r: r["throughput"])
    baseline = all_results[0]

    md = f"""# 동시 요청 처리 실험 결과

실험일: {date_str}
환경: CPU only, ThreadPoolExecutor 시뮬레이션
요청 수: {len(queries_batch)}개 / 측정 단위: 임베딩 검색 (top-k=5)

---

## 결과

| 동시성(workers) | 처리량 | 평균 응답시간 | P95 |
|----------------|--------|------------|-----|
{rows}

---

## 분석

- **단일 요청 기준**: {baseline['mean_ms']}ms 평균
- **최고 처리량**: 동시성 {best['concurrency']}에서 {best['throughput']} req/s
- CPU 임베딩은 torch가 내부적으로 멀티스레드를 사용하므로 단순 ThreadPoolExecutor만으로는
  선형 스케일이 되지 않음. 동시성 증가 시 스레드 경합으로 오히려 개별 응답시간이 늘어날 수 있음.

## 결론

- 현재 CPU 환경에서는 **단일 프로세스 처리**가 합리적 (동시성 증가 대비 처리량 개선 제한적)
- 실제 서비스 확장이 필요하면: **프로세스 분리** (uvicorn workers) 또는 **ONNX Runtime 변환** 후 GIL 우회
- RAG 답변 생성(LLM API 호출)은 I/O bound이므로 `asyncio` 적용 시 동시성 효과가 큼
"""

    out = REPORTS_DIR / f"load_test_{stamp}.md"
    out.write_text(md, encoding="utf-8")
    print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    main()
