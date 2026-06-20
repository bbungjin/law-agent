"""청크 전략 모듈.

1주차: 법령 = 조(條) 단위, 판례 = 판시사항+판결요지 우선
2주차 실험 대상: 조+항 통합, LLM 요약 기반 청크

각 전략은 공통 인터페이스(chunk 메서드)를 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Chunk:
    """검색 인덱스에 들어갈 최소 단위."""

    chunk_id: str       # "{doc_type}_{source_id}_{seq}"
    doc_type: str       # "statute" | "precedent"
    source_id: str      # 법령ID 또는 판례일련번호
    source_name: str    # 법령명 또는 사건명
    text: str           # 검색/임베딩에 쓸 텍스트
    metadata: dict      # 추가 메타 (조번호, 법원명, 선고일자 등)


class ChunkStrategy(Protocol):
    def chunk_statutes(self, statutes: list[dict]) -> list[Chunk]: ...
    def chunk_precedents(self, precedents: list[dict]) -> list[Chunk]: ...


class ArticleLevelStrategy:
    """1주차 기본 전략: 법령은 조 단위, 판례는 판시사항+판결요지 청크."""

    def chunk_statutes(self, statutes: list[dict]) -> list[Chunk]:
        chunks = []
        for i, art in enumerate(statutes):
            chunk_id = f"statute_{art['law_id']}_{i:04d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_type="statute",
                source_id=art["law_id"],
                source_name=art["law_name"],
                text=art["content"],
                metadata={
                    "article_no": art["article_no"],
                    "article_title": art["article_title"],
                    "enforcement_date": art["enforcement_date"],
                },
            ))
        return chunks

    def chunk_precedents(self, precedents: list[dict]) -> list[Chunk]:
        chunks = []
        for prec in precedents:
            # 1순위: 판시사항 + 판결요지 (primary_chunk)
            primary = prec.get("primary_chunk", "").strip()
            if primary:
                chunks.append(Chunk(
                    chunk_id=f"prec_{prec['prec_id']}_primary",
                    doc_type="precedent",
                    source_id=prec["prec_id"],
                    source_name=prec["case_name"],
                    text=primary,
                    metadata={
                        "case_no": prec["case_no"],
                        "court": prec["court"],
                        "judgment_date": prec["judgment_date"],
                        "judgment_type": prec["judgment_type"],
                        "cited_statutes": prec.get("cited_statutes", []),
                    },
                ))
            # 2순위: 판결내용 청크 (primary 없거나 내용이 있는 경우)
            for seq, ct in enumerate(prec.get("content_chunks", [])):
                if not ct.strip():
                    continue
                chunks.append(Chunk(
                    chunk_id=f"prec_{prec['prec_id']}_content_{seq:03d}",
                    doc_type="precedent",
                    source_id=prec["prec_id"],
                    source_name=prec["case_name"],
                    text=ct,
                    metadata={
                        "case_no": prec["case_no"],
                        "court": prec["court"],
                        "judgment_date": prec["judgment_date"],
                        "judgment_type": prec["judgment_type"],
                        "chunk_type": "content",
                        "chunk_seq": seq,
                    },
                ))
        return chunks


class ArticleParagraphStrategy:
    """2주차 실험용: 조+항 통합 청크 (항 단위까지 세분화)."""

    def chunk_statutes(self, statutes: list[dict]) -> list[Chunk]:
        chunks = []
        for art in statutes:
            para_chunks = art.get("chunks", [])
            if not para_chunks:
                # 항 단위 없으면 조 전체를 하나로
                chunks.append(Chunk(
                    chunk_id=f"statute_{art['law_id']}_{art['article_no']}_full",
                    doc_type="statute",
                    source_id=art["law_id"],
                    source_name=art["law_name"],
                    text=art["content"],
                    metadata={"article_no": art["article_no"]},
                ))
            else:
                for seq, para in enumerate(para_chunks):
                    chunks.append(Chunk(
                        chunk_id=f"statute_{art['law_id']}_{art['article_no']}_para{seq}",
                        doc_type="statute",
                        source_id=art["law_id"],
                        source_name=art["law_name"],
                        text=f"{art['article_no']} {para}",
                        metadata={"article_no": art["article_no"], "para_seq": seq},
                    ))
        return chunks

    def chunk_precedents(self, precedents: list[dict]) -> list[Chunk]:
        return ArticleLevelStrategy().chunk_precedents(precedents)


STRATEGIES: dict[str, ChunkStrategy] = {
    "article": ArticleLevelStrategy(),
    "article_paragraph": ArticleParagraphStrategy(),
}


def get_strategy(name: str = "article") -> ChunkStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 청크 전략: {name}. 사용 가능: {list(STRATEGIES)}")
    return STRATEGIES[name]
