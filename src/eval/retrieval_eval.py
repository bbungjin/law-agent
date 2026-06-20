"""검색 성능 평가 — Recall@k, MRR.

QA 셋(data/qa_set.jsonl)의 각 질문에 대해
검색기가 정답 조문/판례를 상위 k개 안에 포함하는지 측정한다.

사용법:
    # 단일 실험
    python src/eval/retrieval_eval.py --retriever embedding --chunk-strategy article
    python src/eval/retrieval_eval.py --retriever bm25
    python src/eval/retrieval_eval.py --retriever hybrid --query-rewrite

    # 2주차 전체 비교 실험은 scripts/run_week2_experiments.py 사용
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

QA_SET_PATH   = ROOT / "data" / "qa_set.jsonl"
INDEX_DIR     = ROOT / "data" / "index"
PROCESSED_DIR = ROOT / "data" / "processed"


# --------------------------------------------------------------------------- #
# 정답 매칭 판단                                                               #
# --------------------------------------------------------------------------- #

def _is_match(chunk_source_name: str, chunk_metadata: dict,
              source_statute: str | None, source_precedent: str | None) -> bool:
    name     = chunk_source_name.lower()
    meta_str = json.dumps(chunk_metadata, ensure_ascii=False).lower()

    if source_statute:
        law_name = source_statute.split()[0].lower()
        if law_name in name:
            parts = source_statute.split()
            if len(parts) > 1:
                article_no = parts[1].replace(",", "").lower()
                if article_no in meta_str or article_no in name:
                    return True
            else:
                return True

    if source_precedent:
        keywords = [w for w in source_precedent.split()
                    if len(w) > 2 and w not in ("판례", "관련", "기반")]
        if any(kw.lower() in name or kw.lower() in meta_str for kw in keywords):
            return True

    return False


# --------------------------------------------------------------------------- #
# 지표 계산                                                                    #
# --------------------------------------------------------------------------- #

def recall_at_k(hit_list: list[bool]) -> float:
    return 1.0 if any(hit_list) else 0.0


def mrr(hit_list: list[bool]) -> float:
    for i, hit in enumerate(hit_list):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------- #
# 검색기 팩토리                                                                #
# --------------------------------------------------------------------------- #

def _load_chunks(chunk_strategy: str) -> list:
    from src.chunking.strategies import get_strategy
    base = "article" if chunk_strategy == "tagged" else chunk_strategy
    strategy   = get_strategy(base)
    statutes   = [json.loads(l) for l in (PROCESSED_DIR / "statutes.jsonl").open(encoding="utf-8")]
    precedents = [json.loads(l) for l in (PROCESSED_DIR / "precedents.jsonl").open(encoding="utf-8")]
    return strategy.chunk_statutes(statutes) + strategy.chunk_precedents(precedents)


def build_retriever(retriever_name: str, chunk_strategy: str, top_k: int):
    """주어진 설정으로 search(query) → [(Chunk, score)] 함수 반환."""

    # tagged 전략은 별도 인덱스 파일 사용
    if chunk_strategy == "tagged":
        index_path = INDEX_DIR / "embedding_index_tagged.npy"
    else:
        index_path = INDEX_DIR / f"embedding_index_{chunk_strategy}.npy"

    if retriever_name == "embedding":
        from src.retrieval.embedding_retriever import EmbeddingRetriever
        r = EmbeddingRetriever()
        r.load_index(index_path)
        return lambda q: r.search(q, top_k=top_k)

    if retriever_name == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        chunks = _load_chunks(chunk_strategy)
        r = BM25Retriever()
        r.build_index(chunks)
        return lambda q: r.search(q, top_k=top_k)

    if retriever_name == "hybrid":
        from src.retrieval.embedding_retriever import EmbeddingRetriever
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        chunks = _load_chunks(chunk_strategy)
        emb = EmbeddingRetriever()
        emb.load_index(index_path)
        bm25 = BM25Retriever()
        bm25.build_index(chunks)
        r = HybridRetriever(emb, bm25)
        return lambda q: r.search(q, top_k=top_k)

    raise ValueError(f"알 수 없는 retriever: {retriever_name}")


# --------------------------------------------------------------------------- #
# 평가 실행                                                                    #
# --------------------------------------------------------------------------- #

def run_eval(
    retriever_name: str = "embedding",
    chunk_strategy: str = "article",
    top_k: int = 5,
    use_query_rewrite: bool = False,
) -> dict:

    if not QA_SET_PATH.exists():
        raise FileNotFoundError(f"QA 셋 없음: {QA_SET_PATH}")
    with open(QA_SET_PATH, encoding="utf-8") as f:
        qa_list = [json.loads(line) for line in f if line.strip()]

    # 인덱스 존재 확인
    index_path = INDEX_DIR / f"embedding_index_{chunk_strategy}.npy"
    if retriever_name in ("embedding", "hybrid") and not index_path.exists():
        raise FileNotFoundError(
            f"인덱스 없음: {index_path}\n"
            f"먼저 실행: python scripts/build_index.py --strategy {chunk_strategy}"
        )

    search_fn = build_retriever(retriever_name, chunk_strategy, top_k)

    # Query Rewriter 준비
    rewriter = None
    if use_query_rewrite:
        from src.retrieval.query_rewriter import CachedQueryRewriter
        rewriter = CachedQueryRewriter()
        print("  [Query Rewriting 활성화]")

    recall_scores, mrr_scores, per_question = [], [], []

    for qa in qa_list:
        question       = qa["question"]
        src_statute    = qa.get("source_statute")
        src_precedent  = qa.get("source_precedent")

        # 쿼리 변환
        if rewriter:
            rewritten, keywords = rewriter.rewrite(question)
            search_query = rewritten
        else:
            search_query = question
            rewritten    = question
            keywords     = []

        results_raw = search_fn(search_query)
        hit_list = [
            _is_match(c.source_name, c.metadata, src_statute, src_precedent)
            for c, _ in results_raw
        ]

        r = recall_at_k(hit_list)
        m = mrr(hit_list)
        recall_scores.append(r)
        mrr_scores.append(m)
        per_question.append({
            "id":           qa["id"],
            "question":     question,
            "rewritten":    rewritten,
            "keywords":     keywords,
            "recall":       r,
            "mrr":          m,
            "top3": [
                f"{c.source_name} {c.metadata.get('article_no', c.metadata.get('case_no', ''))}"
                for c, _ in results_raw[:3]
            ],
        })

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr    = sum(mrr_scores)    / len(mrr_scores)

    return {
        "retriever":       retriever_name,
        "chunk_strategy":  chunk_strategy,
        "query_rewrite":   use_query_rewrite,
        "top_k":           top_k,
        "n_questions":     len(qa_list),
        "recall_at_k":     round(avg_recall, 4),
        "mrr":             round(avg_mrr, 4),
        "per_question":    per_question,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever",      default="embedding",
                        choices=["embedding", "bm25", "hybrid"])
    parser.add_argument("--chunk-strategy", default="article",
                        choices=["article", "article_paragraph", "tagged"])
    parser.add_argument("--top-k",          type=int, default=5)
    parser.add_argument("--query-rewrite",  action="store_true")
    parser.add_argument("--save",           action="store_true")
    args = parser.parse_args()

    label = f"{args.retriever}/{args.chunk_strategy}"
    if args.query_rewrite:
        label += "+QR"
    print(f"\n실험: {label} | top-k: {args.top_k}")
    print("=" * 50)

    result = run_eval(
        retriever_name    = args.retriever,
        chunk_strategy    = args.chunk_strategy,
        top_k             = args.top_k,
        use_query_rewrite = args.query_rewrite,
    )

    print(f"Recall@{args.top_k}: {result['recall_at_k']:.4f}  ({result['recall_at_k']*100:.1f}%)")
    print(f"MRR:          {result['mrr']:.4f}")
    print(f"질문 수:      {result['n_questions']}개")

    print("\n[질문별]")
    for q in result["per_question"]:
        hit = "O" if q["recall"] > 0 else "X"
        rewrite_note = f" -> {q['rewritten'][:35]}" if q["rewritten"] != q["question"] else ""
        print(f"  {hit} [{q['id']}] {q['question'][:35]}{rewrite_note}")

    if args.save:
        import datetime
        reports_dir = ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"eval_{args.retriever}_{args.chunk_strategy}"
        if args.query_rewrite:
            fname += "_qr"
        fname += f"_top{args.top_k}_{stamp}.json"
        out = reports_dir / fname
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
