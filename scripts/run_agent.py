"""에이전트 동작 확인 스크립트.

실행:
    python scripts/run_agent.py
    python scripts/run_agent.py --question "전세금을 못 받았어요. 어떻게 해야 하나요?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DEFAULT_QUESTIONS = [
    "전세금을 못 받았어요. 어떻게 해야 하나요?",
    "회사에서 갑자기 해고당했는데 부당해고인가요?",
    "인터넷 쇼핑몰에서 산 물건을 환불하고 싶은데 거부해요.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", default=None, help="질문 (없으면 기본 예시 실행)")
    args = parser.parse_args()

    from src.agent.agent import LegalAgent

    print("에이전트 초기화 중 (임베딩 모델 로드)...")
    agent = LegalAgent(top_k=5)

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"질문 {i}: {q}")
        print("="*60)

        result = agent.run(q)

        print(f"[선택된 도구]: {result.tool_used}")
        print(f"[검색 쿼리]:   {result.tool_query}")
        print(f"\n[답변]\n{result.answer}")

        if result.sources:
            print(f"\n[출처 ({len(result.sources)}건)]")
            for s in result.sources[:3]:
                print(f"  - [{s['doc_type']}] {s['ref']}")


if __name__ == "__main__":
    main()
