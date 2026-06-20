"""HuggingFace 판례 데이터셋 로더 (API 키 불필요).

데이터셋: joonhok-exo-ai/korean_law_open_data_precedents
  - 국가법령정보 공동활용 전체 판례 (2023년 6월 기준)
  - 필드: 판례일련번호, 사건명, 사건번호, 법원명, 선고일자, 판결유형,
          판시사항, 판결요지, 판결내용
  - 링크: https://huggingface.co/datasets/joonhok-exo-ai/korean_law_open_data_precedents

설치:
  pip install datasets

실행:
  from src.collection.precedent_hf_loader import PrecedentHFLoader
  loader = PrecedentHFLoader(processed_dir=...)
  precs = loader.load_filtered(domain_keywords=["임대차보증금", "부당해고"])
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .precedent_collector import (
    MAX_CONTENT_CHUNK_CHARS,
    OVERLAP_CHARS,
    Precedent,
    _extract_cited_statutes,
    _sliding_window_chunks,
)

HF_DATASET_ID = "joonhok-exo-ai/korean_law_open_data_precedents"

# 수집 대상 도메인 키워드 — 이 중 하나라도 포함된 판례만 수집
DEFAULT_KEYWORDS = [
    "임대차", "전세", "보증금", "주택",        # 주거/임대차
    "임금", "해고", "근로", "퇴직",            # 노동
    "소비자", "청약철회", "환불", "전자상거래",  # 소비자
    "계약금", "손해배상", "채무불이행",          # 계약/채권
]


def _row_to_precedent(row: dict) -> Precedent | None:
    """HuggingFace 행 → Precedent 변환.

    실제 컬럼명 (joonhok-exo-ai/korean_law_open_data_precedents 기준):
      판례일련번호, 사건명, 사건번호, 선고일자, 법원, 법원명,
      판결유형, 판결형태, 판시사항, 판결요지, 판결내용, 참조조문, 참조판례, 전문
    """
    def get(*keys: str) -> str:
        for k in keys:
            v = row.get(k)
            if v is not None and str(v).strip() and str(v) != "None":
                return str(v).strip()
        return ""

    prec_id = get("판례일련번호")
    if not prec_id:
        return None

    issues  = get("판시사항")
    summary = get("판결요지")
    # 판결내용이 None인 경우 전문(전체 판결문) 사용
    content = get("판결내용") or get("전문")

    primary_parts = []
    if issues:
        primary_parts.append(f"[판시사항]\n{issues}")
    if summary:
        primary_parts.append(f"[판결요지]\n{summary}")
    primary_chunk = "\n\n".join(primary_parts)

    content_chunks = _sliding_window_chunks(content) if content else []

    # 참조조문 필드 직접 활용 (정규식 추출보다 정확)
    ref_statutes_raw = get("참조조문")
    if ref_statutes_raw:
        # 쉼표/세미콜론/줄바꿈으로 분리
        cited = [s.strip() for s in re.split(r"[,;\n]", ref_statutes_raw) if s.strip()]
    else:
        cited = _extract_cited_statutes(f"{issues} {summary} {content}")

    return Precedent(
        prec_id=prec_id,
        case_name=get("사건명"),
        case_no=get("사건번호"),
        court=get("법원명", "법원"),
        judgment_date=get("선고일자"),
        judgment_type=get("판결유형", "판결형태"),
        issues=issues,
        summary=summary,
        content=content,
        ref_statutes=get("참조조문"),
        ref_precedents=get("참조판례"),
        primary_chunk=primary_chunk,
        content_chunks=content_chunks,
        cited_statutes=cited,
    )


def _matches_keywords(row: dict, keywords: list[str]) -> bool:
    """판시사항 + 판결요지 + 사건명 중 하나라도 키워드 포함이면 True."""
    text = " ".join(
        str(row.get(f, "") or "")
        for f in ["판시사항", "판결요지", "사건명", "참조조문"]
    )
    return any(kw in text for kw in keywords)


class PrecedentHFLoader:
    """HuggingFace 판례 데이터셋 로더."""

    def __init__(self, processed_dir: Path, dataset_id: str = HF_DATASET_ID):
        self.processed_dir = processed_dir
        self.dataset_id = dataset_id
        processed_dir.mkdir(parents=True, exist_ok=True)

    def load_filtered(
        self,
        domain_keywords: list[str] | None = None,
        max_count: int = 5000,
        split: str = "train",
    ) -> list[Precedent]:
        """키워드 필터링된 판례 목록 반환.

        Args:
            domain_keywords: 이 중 하나라도 포함된 판례만 수집. None이면 DEFAULT_KEYWORDS 사용.
            max_count: 최대 수집 건수 (프로젝트 스코프 제한).
            split: 데이터셋 split 이름.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets 를 먼저 실행하세요.")

        keywords = domain_keywords or DEFAULT_KEYWORDS
        print(f"HuggingFace 데이터셋 로드 중: {self.dataset_id}")
        print("  (streaming 모드 — 전체 다운로드 없이 필터링)")
        print(f"  키워드: {keywords}")

        ds = load_dataset(self.dataset_id, split=split, streaming=True)

        results: list[Precedent] = []
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned % 10000 == 0:
                print(f"  스캔 {scanned:,}건, 수집 {len(results):,}건...")
            if len(results) >= max_count:
                break
            if not _matches_keywords(row, keywords):
                continue
            prec = _row_to_precedent(dict(row))
            if prec:
                results.append(prec)

        print(f"  스캔 완료: {scanned:,}건 중 {len(results):,}건 수집")

        print(f"  필터링 결과: {len(results):,}건")

        out_path = self.processed_dir / "precedents.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for p in results:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        print(f"  저장 완료: {out_path}")

        return results

    def print_sample_columns(self, n: int = 1) -> None:
        """데이터셋 컬럼명과 샘플을 출력해 필드명을 확인한다 (streaming 사용)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets 를 먼저 실행하세요.")

        ds = load_dataset(self.dataset_id, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i == 0:
                print("  컬럼명:", list(row.keys()))
            if i >= n:
                break
            for k, v in row.items():
                val = str(v)[:120] if v and str(v) != "None" else "(없음)"
                print(f"    {k}: {val}")
