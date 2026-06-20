"""법령 수집 및 조(條)/항(項)/호(號) 단위 파싱.

open.law.go.kr XML 구조:

  [법령 검색 — lawSearch.do?target=law]
  <LawSearch>
    <totalCnt>5</totalCnt>
    <page>1</page>
    <numOfRows>20</numOfRows>
    <law>
      <법령명한글>주택임대차보호법</법령명한글>
      <MST>176697</MST>
      <법령ID>법률</법령ID>
      <시행일자>20230901</시행일자>
      <법령구분>법률</법령구분>
    </law>
    ...
  </LawSearch>

  [법령 본문 — lawService.do?target=law&MST=...]
  <법령>
    <기본정보>
      <법령명_한글>주택임대차보호법</법령명_한글>
      <시행일자>20230901</시행일자>
      <MST>176697</MST>
    </기본정보>
    <조문>
      <조문단위>
        <조문번호>1</조문번호>
        <조문제목>목적</조문제목>
        <조문내용>...</조문내용>
        <항>
          <항번호>1</항번호>
          <항내용>...</항내용>
          <호>
            <호번호>1</호번호>
            <호내용>...</호내용>
          </호>
        </항>
      </조문단위>
    </조문>
  </법령>
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .law_api_client import LawAPIClient, _is_html


@dataclass
class StatuteArticle:
    """법령 조(條) 단위 청크."""

    law_id: str            # MST 번호
    law_name: str
    article_no: str        # "제3조"
    article_title: str
    content: str           # 조 전체 텍스트 (항·호 포함)
    enforcement_date: str = ""
    chunks: list[str] = field(default_factory=list)  # 항 단위 (2주차 실험용)


# --------------------------------------------------------------------------- #
# XML 파싱 헬퍼                                                               #
# --------------------------------------------------------------------------- #

def _txt(el: ET.Element, *tags: str) -> str:
    """여러 태그명 중 존재하는 첫 번째 텍스트 값 반환."""
    for tag in tags:
        found = el.findtext(tag)
        if found is not None:
            return found.strip()
    return ""


def _parse_law_search_xml(xml_text: str) -> list[dict]:
    """법령 검색 응답 XML → 법령 기본정보 목록.

    <LawSearch><law><법령일련번호>...</법령일련번호><법령명한글>...</법령명한글></law></LawSearch>

    법령일련번호 = 법령 본문 조회 시 MST 파라미터에 사용하는 값.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[WARN] XML 파싱 오류: {e}")
        return []

    laws = []
    for law_el in root.findall(".//law"):
        # 실제 응답 태그명: 법령일련번호 (MST가 아님)
        mst = _txt(law_el, "법령일련번호", "MST")
        name = _txt(law_el, "법령명한글", "법령명약칭", "법령명")
        enf_date = _txt(law_el, "시행일자")
        if mst:
            laws.append({"mst": mst, "law_name": name, "enforcement_date": enf_date})
    return laws


def _parse_law_detail_xml(xml_text: str) -> tuple[str, str, str, list[StatuteArticle]]:
    """법령 본문 XML → (mst, law_name, enforcement_date, articles).

    <법령><기본정보>...</기본정보><조문><조문단위>...</조문단위></조문></법령>
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[WARN] XML 파싱 오류: {e}")
        return "", "", "", []

    # 기본정보
    basic = root.find("기본정보")
    if basic is not None:
        mst = _txt(basic, "법령일련번호", "MST")
        law_name = _txt(basic, "법령명_한글", "법령명한글", "법령명")
        enf_date = _txt(basic, "시행일자")
    else:
        mst = _txt(root, "법령일련번호", "MST")
        law_name = _txt(root, "법령명_한글", "법령명한글")
        enf_date = _txt(root, "시행일자")

    articles: list[StatuteArticle] = []
    for jo_el in root.findall(".//조문단위"):
        art_no_raw = _txt(jo_el, "조문번호")
        if not art_no_raw:
            continue

        art_title = _txt(jo_el, "조문제목")
        jo_content = _txt(jo_el, "조문내용")

        # "제N조" 형식 변환
        if art_no_raw.isdigit():
            display_no = f"제{int(art_no_raw)}조"
        else:
            display_no = art_no_raw

        parts: list[str] = []
        para_chunks: list[str] = []

        if jo_content:
            parts.append(jo_content)

        for hang_el in jo_el.findall("항"):
            hang_no = _txt(hang_el, "항번호")
            hang_content = _txt(hang_el, "항내용")
            if not hang_content:
                continue

            line = f"제{hang_no}항 {hang_content}"
            ho_lines = []
            for ho_el in hang_el.findall("호"):
                ho_no = _txt(ho_el, "호번호")
                ho_content = _txt(ho_el, "호내용")
                if ho_content:
                    ho_lines.append(f"  {ho_no}. {ho_content}")
            if ho_lines:
                line += "\n" + "\n".join(ho_lines)
            parts.append(line)
            para_chunks.append(line)

        if not parts:
            continue

        header = display_no
        if art_title:
            header += f"({art_title})"
        content = f"[{law_name}] {header}\n" + "\n".join(parts)

        articles.append(StatuteArticle(
            law_id=mst,
            law_name=law_name,
            article_no=display_no,
            article_title=art_title,
            content=content,
            enforcement_date=enf_date,
            chunks=para_chunks,
        ))

    return mst, law_name, enf_date, articles


# --------------------------------------------------------------------------- #
# 수집기                                                                       #
# --------------------------------------------------------------------------- #

class StatuteCollector:
    def __init__(self, client: LawAPIClient, raw_dir: Path, processed_dir: Path):
        self.client = client
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

    def find_mst(self, law_name: str) -> str | None:
        """법령명으로 검색해 MST 번호 반환 (정확 일치 우선)."""
        xml_text = self.client.search_laws(law_name, page=1)
        laws = _parse_law_search_xml(xml_text)
        if not laws:
            return None
        for law in laws:
            if law["law_name"] == law_name:
                return law["mst"]
        return laws[0]["mst"]

    def collect_statute(self, law_name: str, mst: str | None = None) -> list[StatuteArticle]:
        if mst is None:
            mst = self.find_mst(law_name)
        if not mst:
            print(f"[WARN] 법령일련번호 조회 실패: {law_name}")
            return []

        raw_path = self.raw_dir / f"law_{mst}.xml"
        if raw_path.exists():
            xml_text = raw_path.read_text(encoding="utf-8")
            if _is_html(xml_text):
                print(f"  [SKIP] 캐시가 HTML — 재요청: {law_name}")
                raw_path.unlink()
                xml_text = None
        else:
            xml_text = None

        if xml_text is None:
            try:
                xml_text = self.client.get_law_detail(mst)
            except ValueError as e:
                print(f"  [WARN] {e}")
                return []
            raw_path.write_text(xml_text, encoding="utf-8")

        _, resolved_name, _, articles = _parse_law_detail_xml(xml_text)
        name = resolved_name or law_name
        print(f"  [{name}] {len(articles)}개 조문")
        return articles

    def collect_all(self, target_laws: list[dict]) -> list[StatuteArticle]:
        all_articles: list[StatuteArticle] = []
        for law_info in target_laws:
            name = law_info["name"]
            mst = law_info.get("mst") or law_info.get("law_id")
            print(f"수집: {name}")
            articles = self.collect_statute(name, mst=mst)
            all_articles.extend(articles)

        out_path = self.processed_dir / "statutes.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for art in all_articles:
                f.write(json.dumps(asdict(art), ensure_ascii=False) + "\n")
        print(f"\n총 {len(all_articles)}개 조문 → {out_path}")
        return all_articles
