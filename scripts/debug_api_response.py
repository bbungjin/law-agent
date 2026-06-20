"""API 응답 XML 구조 확인 스크립트.

LAW_OC 설정 직후 실행해서 실제 XML 태그명을 확인한다.
파서(statute_collector.py / precedent_collector.py)의 태그명이
실제 응답과 다르면 이 출력을 보고 수정한다.

    python scripts/debug_api_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import xml.etree.ElementTree as ET
from src.collection.law_api_client import LawAPIClient


def pretty_print_xml(xml_text: str, max_chars: int = 4000) -> None:
    try:
        root = ET.fromstring(xml_text)
        ET.indent(root, space="  ")
        text = ET.tostring(root, encoding="unicode")
        print(text[:max_chars])
        if len(text) > max_chars:
            print(f"\n... (총 {len(text)}자, 처음 {max_chars}자만 출력)")
    except ET.ParseError:
        print(xml_text[:max_chars])


def main() -> None:
    client = LawAPIClient()

    print("=" * 60)
    print("[1] 법령 검색 — 주택임대차보호법")
    print("=" * 60)
    xml = client.search_laws("주택임대차보호법", page=1)
    pretty_print_xml(xml)

    # 첫 번째 MST 추출 후 본문 조회
    try:
        root = ET.fromstring(xml)
        mst = root.findtext(".//MST") or root.findtext(".//mst")
        if mst:
            print("\n" + "=" * 60)
            print(f"[2] 법령 본문 조회 — MST: {mst}")
            print("=" * 60)
            detail_xml = client.get_law_detail(mst.strip())
            pretty_print_xml(detail_xml)
    except Exception as e:
        print(f"법령 본문 조회 실패: {e}")

    print("\n" + "=" * 60)
    print("[3] 판례 검색 — '임대차보증금반환'")
    print("=" * 60)
    xml = client.search_precedents("임대차보증금반환", page=1)
    pretty_print_xml(xml)

    # 첫 번째 판례 ID 추출 후 본문 조회
    try:
        root = ET.fromstring(xml)
        prec_id = root.findtext(".//판례일련번호")
        if prec_id:
            print("\n" + "=" * 60)
            print(f"[4] 판례 본문 조회 — ID: {prec_id}")
            print("=" * 60)
            detail_xml = client.get_precedent_detail(prec_id.strip())
            pretty_print_xml(detail_xml)
    except Exception as e:
        print(f"판례 본문 조회 실패: {e}")


if __name__ == "__main__":
    main()
