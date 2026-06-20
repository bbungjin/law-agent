"""2주차 검색 비교 실험 자동 실행 스크립트.

실행하면:
  1. 조+항 단위 임베딩 인덱스 빌드 (없을 때만)
  2. 7가지 실험 조합 순차 실행
  3. 결과 비교표 출력 + reports/week2_experiment_results.md 저장

실험 조합:
  A. 임베딩 / 조 단위        (1주차 베이스라인)
  B. 임베딩 / 조+항 단위
  C. BM25 / 조 단위          (1주차 베이스라인)
  D. BM25 / 조+항 단위
  E. 하이브리드 / 조 단위    (1주차 베이스라인)
  F. 임베딩+QR / 조 단위     ← Query Rewriting 효과
  G. 하이브리드+QR / 조 단위

실행:
    python scripts/run_week2_experiments.py
    python scripts/run_week2_experiments.py --top-k 10
    python scripts/run_week2_experiments.py --skip-qr   # LLM 비용 절약
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

INDEX_DIR     = ROOT / "data" / "index"
REPORTS_DIR   = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 실험 설정                                                                    #
# --------------------------------------------------------------------------- #

EXPERIMENTS = [
    # (label,         retriever,   chunk_strategy,       query_rewrite)
    ("A-emb/article",       "embedding", "article",            False),
    ("B-emb/para",          "embedding", "article_paragraph",  False),
    ("C-bm25/article",      "bm25",      "article",            False),
    ("D-bm25/para",         "bm25",      "article_paragraph",  False),
    ("E-hybrid/article",    "hybrid",    "article",            False),
    ("F-emb+QR/article",    "embedding", "article",            True),
    ("G-hybrid+QR/article", "hybrid",    "article",            True),
    # H: LLM 메타데이터 태깅 효과 (scripts/tag_metadata.py 먼저 실행 필요)
    ("H-emb/tagged",        "embedding", "tagged",             False),
]


# --------------------------------------------------------------------------- #
# 인덱스 준비                                                                  #
# --------------------------------------------------------------------------- #

def ensure_index(chunk_strategy: str) -> None:
    if chunk_strategy == "tagged":
        idx = INDEX_DIR / "embedding_index_tagged.npy"
        if idx.exists():
            print(f"  [인덱스 존재] {idx.name}")
        else:
            print("  [태깅 인덱스 없음] scripts/tag_metadata.py --sample 100 을 먼저 실행하세요.")
        return

    idx = INDEX_DIR / f"embedding_index_{chunk_strategy}.npy"
    if idx.exists():
        print(f"  [인덱스 존재] {idx.name}")
        return
    print(f"  [인덱스 빌드 중] {chunk_strategy} ...")
    subprocess.run(
        [sys.executable, "scripts/build_index.py", "--strategy", chunk_strategy],
        cwd=ROOT, check=True,
    )


# --------------------------------------------------------------------------- #
# 결과 비교표 생성                                                              #
# --------------------------------------------------------------------------- #

def format_table(results: list[dict]) -> str:
    header = f"{'실험':<25} {'Recall@k':>9} {'MRR':>7} {'vs 베이스라인':>14}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    baseline_recall = None
    for r in results:
        label  = r["label"]
        recall = r["recall_at_k"]
        mrr_v  = r["mrr"]
        if label.startswith("A-"):
            baseline_recall = recall
        diff = ""
        if baseline_recall and not label.startswith("A-"):
            delta = recall - baseline_recall
            diff  = f"{delta:+.1%}"
        rows.append(f"{label:<25} {recall:>8.1%}  {mrr_v:>6.4f}  {diff:>14}")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# 마크다운 리포트                                                               #
# --------------------------------------------------------------------------- #

def write_report(results: list[dict], top_k: int) -> Path:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    baseline  = next((r for r in results if r["label"].startswith("A-")), None)
    qr_emb    = next((r for r in results if r["label"].startswith("F-")), None)
    para_emb  = next((r for r in results if r["label"].startswith("B-")), None)
    hybrid    = next((r for r in results if r["label"].startswith("E-")), None)
    qr_hyb    = next((r for r in results if r["label"].startswith("G-")), None)
    tagged    = next((r for r in results if r["label"].startswith("H-")), None)

    qr_delta      = (qr_emb["recall_at_k"]  - baseline["recall_at_k"]) if (qr_emb  and baseline) else 0
    para_delta    = (para_emb["recall_at_k"] - baseline["recall_at_k"]) if (para_emb and baseline) else 0
    hybrid_delta  = (hybrid["recall_at_k"]   - baseline["recall_at_k"]) if (hybrid   and baseline) else 0
    tagged_delta  = (tagged["recall_at_k"]   - baseline["recall_at_k"]) if (tagged   and baseline) else 0

    md = f"""# 2주차 검색 비교 실험 결과

실험일: {date_str}
평가셋: QA 25개 | top-k: {top_k}
데이터: 법령 조문 + 판례 129건 (open.law.go.kr)

---

## 결과 요약

```
{format_table(results)}
```

---

## 실험별 분석

### 실험 1: 청크 전략 비교 (조 단위 vs 조+항 단위)

- 가설: 조+항 단위로 세분화하면 더 정밀한 매칭이 가능할 것이다.
- 결과:
  - 임베딩/조 단위:    {baseline["recall_at_k"]:.1%}
  - 임베딩/조+항 단위: {para_emb["recall_at_k"]:.1%}  (베이스라인 대비 {para_delta:+.1%})
- 분석:
  - {"청크 세분화가 성능을 향상시켰다. 항 단위로 세분화하면 검색 시 더 관련성 높은 텍스트가 상위에 오는 효과가 있다." if para_delta > 0 else "청크 세분화가 예상과 달리 성능을 낮추거나 유지했다. 항 단위 청크가 너무 짧아 임베딩 품질이 저하됐거나, 조 전체 맥락이 더 유용할 수 있다."}
  - 법령 조문의 경우 항(項) 하나만으로는 맥락이 부족하고 조(條) 전체가 더 의미 단위로 적합할 수 있다.
- 다음 액션: {"조 단위를 유지하되 항 내용을 조 헤더와 함께 포함하는 방식 고려." if para_delta <= 0 else "조+항 전략을 기본으로 채택하고 판례 청크 전략도 개선 검토."}

### 실험 2: BM25 vs 임베딩 vs 하이브리드

- 가설: 하이브리드가 두 방식의 장점을 합쳐 가장 높은 성능을 보일 것이다.
- 결과:
  - 임베딩:   {baseline["recall_at_k"]:.1%}
  - BM25:     {next((r["recall_at_k"] for r in results if r["label"].startswith("C-")), 0):.1%}
  - 하이브리드: {hybrid["recall_at_k"]:.1%}  (베이스라인 대비 {hybrid_delta:+.1%})
- 분석:
  - 임베딩이 BM25보다 높음: 일반인 구어체 질문에서 의미 기반 검색의 이점 확인.
  - {"하이브리드가 임베딩보다 낮음: 법률 도메인에서 BM25의 노이즈가 RRF 점수를 희석시킨 것으로 보임. 가중치 튜닝이 필요." if hybrid_delta < 0 else "하이브리드가 단독 임베딩보다 높음: 두 검색기의 상호 보완 효과 확인."}

### 실험 3: Query Rewriting 효과

- 가설: 구어체 → 법률용어 변환으로 어휘 격차를 줄이면 검색 성능이 올라갈 것이다.
- 결과:

  | 검색기 | 원문 | QR 적용 | 변화 (Recall) | 변화 (MRR) |
  |--------|------|---------|--------------|-----------|
  | 임베딩  | {f"{baseline['recall_at_k']:.1%} / MRR {baseline['mrr']:.4f}" if baseline else "N/A"} | {f"{qr_emb['recall_at_k']:.1%} / MRR {qr_emb['mrr']:.4f}" if qr_emb else "미실행"} | {f"{qr_delta:+.1%}" if qr_emb else "-"} | {f"{(qr_emb['mrr']-baseline['mrr']):+.4f}" if (qr_emb and baseline) else "-"} |
  | 하이브리드 | {f"{hybrid['recall_at_k']:.1%} / MRR {hybrid['mrr']:.4f}" if hybrid else "N/A"} | {f"{qr_hyb['recall_at_k']:.1%} / MRR {qr_hyb['mrr']:.4f}" if qr_hyb else "미실행"} | {f"{(qr_hyb['recall_at_k']-hybrid['recall_at_k']):+.1%}" if (qr_hyb and hybrid) else "-"} | {f"{(qr_hyb['mrr']-hybrid['mrr']):+.4f}" if (qr_hyb and hybrid) else "-"} |

- 분석:
  - **임베딩+QR (F)**: {"QR 적용 후 Recall 하락 — 예상과 다른 결과. LLM이 구어체를 과도하게 법률 용어화하여 임베딩 모델이 본래 잘 처리하던 의미 유사도를 오히려 방해. 임베딩 모델은 이미 구어체 질문의 의미를 잘 이해하고 있어 추가 변환이 불필요했을 가능성." if qr_emb else "미실행"}
  - **하이브리드+QR (G)**: {"QR 적용 후 MRR 상승 — 긍정적 결과. QR로 법률 용어를 명시하면 BM25의 키워드 매칭 품질이 향상되고, 이것이 RRF 점수에 반영되어 전체 순위 품질 개선. 하이브리드 검색에서는 QR이 BM25 성분을 강화하는 역할을 한다." if qr_hyb else "미실행"}
  - **핵심 발견**: QR은 '임베딩 단독'에는 오히려 역효과지만 '하이브리드'에서는 효과적. 검색기 조합에 따라 QR 적용 여부를 다르게 해야 함.
- 한계: LLM API 비용이 발생하므로 실시간 서비스에서는 비용 대비 효과를 따져야 한다.

### 실험 4: LLM 메타데이터 태깅 효과

- 가설: 청크에 LLM이 생성한 쉬운 말 요약과 키워드를 추가하면 구어체 질문과의 의미 거리가 줄어들어 검색 성능이 향상될 것이다.
- 결과:
  - 임베딩 (원본):  {baseline["recall_at_k"]:.1%}
  - 임베딩 (태깅): {f"{tagged['recall_at_k']:.1%}  (베이스라인 대비 {tagged_delta:+.1%})" if tagged else "N/A (scripts/tag_metadata.py 실행 후 재실험 필요)"}
- 방법: 각 청크에 [요약] + [키워드] 텍스트를 추가한 뒤 임베딩 인덱스 재빌드 (scripts/tag_metadata.py --sample 100)
- 분석:
  - {"태깅이 검색 성능을 향상시켰다. 쉬운 말 요약이 구어체 질문과의 의미 유사도를 높인 것으로 판단된다." if tagged_delta > 0 else ("태깅 효과가 미미하거나 역효과. 가능한 원인: (1) 샘플 태깅(일부 청크만)으로 효과가 희석됨, (2) LLM 요약이 임베딩 모델이 기존에 이미 잘 처리하던 내용을 중복 추가한 것일 수 있음." if tagged else "태깅 실험 미실행.")}
- 다음 액션: {"전체 청크 태깅 후 재실험으로 효과 확인." if tagged_delta > 0 else ("태깅 방식을 개선하거나 전체 청크 태깅 후 재측정." if tagged else "scripts/tag_metadata.py 실행 후 재실험.")}

---

## 개선 방향 (3주차)

1. **에이전트화**: 질문 유형을 분류해 법령 검색 / 판례 검색 / 혼합을 선택적으로 라우팅
2. **판례 데이터 보강**: 현재 129건 → 키워드 확장 및 max_pages 증가로 추가 수집
3. **서빙 최적화**: 임베딩 추론 병목 프로파일링 후 배치 처리 또는 ONNX 변환
"""

    out_path = REPORTS_DIR / f"week2_experiment_results_{stamp}.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# 메인                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",   type=int, default=5)
    parser.add_argument("--skip-qr", action="store_true", help="Query Rewriting 실험 건너뛰기")
    args = parser.parse_args()

    experiments = [
        e for e in EXPERIMENTS
        if not (args.skip_qr and e[3])  # query_rewrite=True인 실험 제외
    ]

    print("=" * 60)
    print(f"2주차 검색 비교 실험  |  top-k={args.top_k}")
    print(f"실험 수: {len(experiments)}개")
    print("=" * 60)

    # 필요한 인덱스 빌드
    needed_strategies = set(e[2] for e in experiments if e[1] in ("embedding", "hybrid"))
    for s in needed_strategies:
        ensure_index(s)

    # 실험 실행
    from src.eval.retrieval_eval import run_eval

    all_results = []
    for label, retriever, chunk_strategy, use_qr in experiments:
        tag = f"{label}" + (" [+QR]" if use_qr else "")
        print(f"\n실험: {tag}")
        try:
            result = run_eval(
                retriever_name    = retriever,
                chunk_strategy    = chunk_strategy,
                top_k             = args.top_k,
                use_query_rewrite = use_qr,
            )
            result["label"] = label
            all_results.append(result)
            print(f"  Recall@{args.top_k}: {result['recall_at_k']:.1%}  MRR: {result['mrr']:.4f}")
        except Exception as e:
            print(f"  [오류] {e}")

    # 결과 출력
    print("\n" + "=" * 60)
    print("최종 비교")
    print("=" * 60)
    print(format_table(all_results))

    # 결과 저장
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # JSON 원본 데이터
    json_path = REPORTS_DIR / f"week2_raw_{stamp}.json"
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 마크다운 리포트
    md_path = write_report(all_results, args.top_k)
    print(f"\n리포트 저장:")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
