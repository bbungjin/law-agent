"""임베딩 인덱스 구축.

실행:
    python scripts/build_index.py                        # 기본 (조 단위)
    python scripts/build_index.py --strategy article_paragraph  # 조+항 단위
    python scripts/build_index.py --strategy all         # 모든 전략 빌드

전제: scripts/collect_data.py 먼저 실행 필요.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking.strategies import STRATEGIES, get_strategy
from src.retrieval.embedding_retriever import EmbeddingRetriever

PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR     = ROOT / "data" / "index"
INDEX_DIR.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_for_strategy(strategy_name: str) -> None:
    strategy = get_strategy(strategy_name)

    statutes   = load_jsonl(PROCESSED_DIR / "statutes.jsonl")
    precedents = load_jsonl(PROCESSED_DIR / "precedents.jsonl")

    statute_chunks = strategy.chunk_statutes(statutes)
    prec_chunks    = strategy.chunk_precedents(precedents)
    all_chunks     = statute_chunks + prec_chunks

    print(f"\n[{strategy_name}] 법령 {len(statute_chunks)}개 + 판례 {len(prec_chunks)}개 = {len(all_chunks)}개 청크")

    if not all_chunks:
        print("청크 없음. collect_data.py를 먼저 실행하세요.")
        return

    index_path = INDEX_DIR / f"embedding_index_{strategy_name}.npy"
    retriever  = EmbeddingRetriever()
    retriever.build_index(all_chunks, cache_path=index_path)
    print(f"인덱스 저장: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default="article",
        choices=list(STRATEGIES) + ["all"],
        help="청크 전략 선택 (기본: article)",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        for name in STRATEGIES:
            build_for_strategy(name)
    else:
        build_for_strategy(args.strategy)

    # 기본 인덱스(article)를 embedding_index.npy로도 복사 (앱 호환성 유지)
    default_src = INDEX_DIR / "embedding_index_article.npy"
    default_dst = INDEX_DIR / "embedding_index.npy"
    if default_src.exists() and not default_dst.exists():
        import shutil
        shutil.copy2(default_src, default_dst)
        print(f"기본 인덱스 복사: {default_dst}")

    print("\n임베딩 인덱스 구축 완료!")


if __name__ == "__main__":
    main()
