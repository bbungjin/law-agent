"""LLM 메타데이터 태깅 스크립트.

전체 청크(또는 일부 샘플)에 LLM이 생성한 요약과 키워드를 추가하고,
태깅된 텍스트로 임베딩 인덱스를 빌드하여 검색 성능 변화를 측정한다.

실행:
    python scripts/tag_metadata.py --sample 100    # 100개 샘플만 태깅
    python scripts/tag_metadata.py                 # 전체 태깅 (비용 주의)

산출물:
    data/index/metadata_tags.json            태깅 결과 캐시
    data/index/embedding_index_tagged.npy    태깅 기반 임베딩 인덱스
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR     = ROOT / "data" / "index"
INDEX_DIR.mkdir(exist_ok=True)

TAGS_CACHE = INDEX_DIR / "metadata_tags.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",   type=int, default=None,
                        help="태깅할 청크 수 제한 (기본: 전체)")
    parser.add_argument("--strategy", default="article",
                        choices=["article", "article_paragraph"],
                        help="기반 청크 전략 (기본: article)")
    parser.add_argument("--skip-tag", action="store_true",
                        help="태깅 생략, 기존 캐시만 사용해 인덱스 빌드")
    args = parser.parse_args()

    from src.chunking.strategies import get_strategy
    from src.chunking.metadata_tagger import MetadataTagger, enrich_chunk_text
    from src.retrieval.embedding_retriever import EmbeddingRetriever

    # 청크 로드
    statutes   = load_jsonl(PROCESSED_DIR / "statutes.jsonl")
    precedents = load_jsonl(PROCESSED_DIR / "precedents.jsonl")
    strategy   = get_strategy(args.strategy)
    all_chunks = strategy.chunk_statutes(statutes) + strategy.chunk_precedents(precedents)

    print(f"전체 청크: {len(all_chunks)}개")

    # 기존 태그 캐시 로드
    existing_tags: dict[str, dict] = {}
    if TAGS_CACHE.exists():
        raw = json.loads(TAGS_CACHE.read_text(encoding="utf-8"))
        existing_tags = {t["chunk_id"]: t for t in raw}
        print(f"기존 태그 캐시: {len(existing_tags)}개")

    if not args.skip_tag:
        # 아직 태깅되지 않은 청크만 선택
        untagged = [c for c in all_chunks if c.chunk_id not in existing_tags]
        if args.sample:
            # 법령/판례 균형 샘플링
            statute_untagged  = [c for c in untagged if c.doc_type == "statute"][:args.sample // 2]
            prec_untagged     = [c for c in untagged if c.doc_type == "precedent"][:args.sample // 2]
            targets           = statute_untagged + prec_untagged
        else:
            targets = untagged

        if targets:
            print(f"\nLLM 태깅 시작: {len(targets)}개 청크")
            print("예상 비용: ~$0.01-0.05 (gpt-4o-mini 기준)")
            tagger  = MetadataTagger()
            new_tags = tagger.tag_chunks(targets)

            # 캐시 업데이트
            for t in new_tags:
                existing_tags[t["chunk_id"]] = t
            TAGS_CACHE.write_text(
                json.dumps(list(existing_tags.values()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n태그 캐시 저장: {TAGS_CACHE} ({len(existing_tags)}개)")
        else:
            print("모든 청크가 이미 태깅됨")

    # 태깅된 청크로 인덱스 빌드
    tagged_chunks = []
    skipped = 0
    for c in all_chunks:
        tag = existing_tags.get(c.chunk_id)
        if tag:
            tagged_chunks.append(enrich_chunk_text(c, tag))
        else:
            tagged_chunks.append(c)  # 태그 없으면 원본 사용
            skipped += 1

    print(f"\n인덱스 빌드: {len(tagged_chunks)}개 청크 (태그 없음: {skipped}개)")
    index_path = INDEX_DIR / "embedding_index_tagged.npy"
    retriever  = EmbeddingRetriever()
    retriever.build_index(tagged_chunks, cache_path=index_path)
    print(f"인덱스 저장: {index_path}")


if __name__ == "__main__":
    main()
