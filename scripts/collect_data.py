"""데이터 수집 스크립트 — open.law.go.kr 국가법령정보 공동활용 API 사용.

===== 실행 전 준비 =====
1. https://open.law.go.kr 회원가입 → API 활용신청 (승인 1~2일)
2. .env 파일에 설정:
     LAW_OC=가입한이메일주소

===== 실행 =====
  python scripts/collect_data.py
  python scripts/build_index.py

===== 결과 =====
  data/raw/law_{MST}.xml         - 법령 원본 XML (gitignore)
  data/raw/prec_{ID}.xml         - 판례 원본 XML (gitignore)
  data/processed/statutes.jsonl
  data/processed/precedents.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.collection.law_api_client import LawAPIClient
from src.collection.statute_collector import StatuteCollector
from src.collection.precedent_collector import PrecedentCollector

RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET_YAML   = ROOT / "data" / "target_laws.yaml"


def load_target_laws() -> tuple[list[dict], list[str]]:
    with TARGET_YAML.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    laws = config.get("target_laws", [])
    keywords = config.get("precedent_keywords", [])
    return laws, keywords


def main() -> None:
    client = LawAPIClient()

    laws, keywords = load_target_laws()

    # ── 법령 수집 ──────────────────────────────────────────────
    print("=" * 55)
    print("법령 수집 — open.law.go.kr API")
    print("=" * 55)
    statute_collector = StatuteCollector(client, RAW_DIR, PROCESSED_DIR)
    statutes = statute_collector.collect_all(laws)

    # ── 판례 수집 ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("판례 수집 — open.law.go.kr API")
    print("=" * 55)
    prec_collector = PrecedentCollector(client, RAW_DIR, PROCESSED_DIR)
    precedents = prec_collector.collect_by_keywords(keywords, max_pages_per_keyword=3)

    print("\n" + "=" * 55)
    print("수집 완료")
    print(f"  법령 조문: {len(statutes):,}개")
    print(f"  판례:      {len(precedents):,}건")
    print("=" * 55)
    print("다음 단계: python scripts/build_index.py")


if __name__ == "__main__":
    main()
