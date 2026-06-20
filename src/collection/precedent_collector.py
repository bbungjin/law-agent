"""판례 수집 및 필드 파싱.

open.law.go.kr XML 구조:

  [판례 검색 — lawSearch.do?target=prec]
  <PrecSearch>
    <totalCnt>100</totalCnt>
    <page>1</page>
    <numOfRows>20</numOfRows>
    <prec>
      <판례일련번호>12345</판례일련번호>
      <사건명>전세금반환</사건명>
      <사건번호>2020다12345</사건번호>
      <선고일자>20201201</선고일자>
      <법원명>대법원</법원명>
      <사건종류명>민사</사건종류명>
      <판결유형>판결</판결유형>
    </prec>
    ...
  </PrecSearch>

  [판례 본문 — lawService.do?target=prec&MST=...]
  <PrecService>
    <판례일련번호>12345</판례일련번호>
    <사건명>전세금반환</사건명>
    <사건번호>2020다12345</사건번호>
    <선고일자>20201201</선고일자>
    <법원명>대법원</법원명>
    <판결유형>판결</판결유형>
    <판시사항>...</판시사항>
    <판결요지>...</판결요지>
    <참조조문>...</참조조문>
    <참조판례>...</참조판례>
    <판결내용>...</판결내용>
  </PrecService>
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .law_api_client import LawAPIClient, _is_html, _is_error_response
from .statute_collector import _txt

MAX_CONTENT_CHUNK_CHARS = 800
OVERLAP_CHARS = 100


@dataclass
class Precedent:
    prec_id: str
    case_name: str
    case_no: str
    court: str
    judgment_date: str
    judgment_type: str
    issues: str             # 판시사항
    summary: str            # 판결요지
    content: str            # 판결내용
    ref_statutes: str = ""  # 참조조문
    ref_precedents: str = ""  # 참조판례
    primary_chunk: str = ""
    content_chunks: list[str] = field(default_factory=list)
    cited_statutes: list[str] = field(default_factory=list)


def _extract_cited_statutes(text: str) -> list[str]:
    pattern = r"[가-힣\s]{2,20}법\s*제\s*\d+조(?:의\d+)?(?:\s*제\s*\d+항)?(?:\s*제\s*\d+호)?"
    matches = re.findall(pattern, text)
    return list(dict.fromkeys(m.strip() for m in matches))


def _sliding_window_chunks(text: str, max_chars: int = MAX_CONTENT_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _parse_prec_search_xml(xml_text: str) -> list[dict]:
    """판례 검색 XML → 판례 기본정보 목록."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[WARN] XML 파싱 오류: {e}")
        return []

    results = []
    for prec_el in root.findall(".//prec"):
        prec_id = _txt(prec_el, "판례일련번호")
        if not prec_id:
            continue
        results.append({
            "prec_id": prec_id,
            "case_name": _txt(prec_el, "사건명"),
            "case_no": _txt(prec_el, "사건번호"),
            "court": _txt(prec_el, "법원명"),
            "judgment_date": _txt(prec_el, "선고일자"),
            "judgment_type": _txt(prec_el, "판결유형"),
        })
    return results


def _parse_prec_detail_xml(xml_text: str) -> Precedent | None:
    """판례 본문 XML → Precedent.

    <PrecService> 루트 또는 단일 item 구조 모두 지원.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[WARN] XML 파싱 오류: {e}")
        return None

    # <PrecService> 루트이거나 내부 item 구조
    el = root if root.tag == "PrecService" else root.find(".//PrecService")
    if el is None:
        # fallback: item 구조
        el = root.find(".//item")
    if el is None:
        el = root

    # 본문 조회 응답의 ID 태그는 검색 응답(판례일련번호)과 다름
    prec_id = _txt(el, "판례정보일련번호", "판례일련번호")
    if not prec_id:
        return None

    issues = _txt(el, "판시사항")
    summary = _txt(el, "판결요지")
    # 본문 조회 응답 태그: 판례내용 (검색 응답 또는 구버전: 판결내용)
    content = _txt(el, "판례내용", "판결내용")
    ref_statutes = _txt(el, "참조조문")
    ref_precedents = _txt(el, "참조판례")

    primary_parts = []
    if issues:
        primary_parts.append(f"[판시사항]\n{issues}")
    if summary:
        primary_parts.append(f"[판결요지]\n{summary}")
    primary_chunk = "\n\n".join(primary_parts)

    content_chunks = _sliding_window_chunks(content) if content else []
    cited = _extract_cited_statutes(f"{issues} {summary} {content} {ref_statutes}")

    return Precedent(
        prec_id=prec_id,
        case_name=_txt(el, "사건명"),
        case_no=_txt(el, "사건번호"),
        court=_txt(el, "법원명"),
        judgment_date=_txt(el, "선고일자"),
        judgment_type=_txt(el, "판결유형"),
        issues=issues,
        summary=summary,
        content=content,
        ref_statutes=ref_statutes,
        ref_precedents=ref_precedents,
        primary_chunk=primary_chunk,
        content_chunks=content_chunks,
        cited_statutes=cited,
    )


class PrecedentCollector:
    def __init__(self, client: LawAPIClient, raw_dir: Path, processed_dir: Path):
        self.client = client
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

    def search_precedents(self, keyword: str, max_pages: int = 3) -> list[dict]:
        results = []
        for page in range(1, max_pages + 1):
            xml_text = self.client.search_precedents(keyword, page=page)
            items = _parse_prec_search_xml(xml_text)
            if not items:
                break
            results.extend(items)
        return results

    def collect_precedent(self, prec_id: str) -> Precedent | None:
        raw_path = self.raw_dir / f"prec_{prec_id}.xml"
        if raw_path.exists():
            xml_text = raw_path.read_text(encoding="utf-8")
            # 오류 응답(HTML 또는 "일치하는 판례 없음" XML) 캐시 무효화
            if _is_error_response(xml_text):
                raw_path.unlink()
                xml_text = None
        else:
            xml_text = None

        if xml_text is None:
            try:
                xml_text = self.client.get_precedent_detail(prec_id)
            except ValueError as e:
                print(f"  [WARN] {e}")
                return None
            # 오류 응답은 저장하지 않음
            if not _is_error_response(xml_text):
                raw_path.write_text(xml_text, encoding="utf-8")

        return _parse_prec_detail_xml(xml_text)

    def collect_by_keywords(self, keywords: list[str], max_pages_per_keyword: int = 3) -> list[Precedent]:
        seen_ids: set[str] = set()
        all_precs: list[Precedent] = []

        for kw in keywords:
            print(f"판례 검색: '{kw}'")
            meta_list = self.search_precedents(kw, max_pages=max_pages_per_keyword)
            print(f"  {len(meta_list)}건 발견")
            for meta in meta_list:
                pid = meta["prec_id"]
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                prec = self.collect_precedent(pid)
                if prec:
                    all_precs.append(prec)

        out_path = self.processed_dir / "precedents.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for p in all_precs:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        print(f"\n총 {len(all_precs)}개 판례 → {out_path}")
        return all_precs
