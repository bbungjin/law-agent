"""LLM 기반 메타데이터 태깅.

각 청크에 LLM이 생성한 키워드와 쉬운 말 요약을 추가하여
검색 텍스트를 보강한다.

사용법:
    python scripts/tag_metadata.py            # 전체 청크 태깅
    python scripts/tag_metadata.py --sample 50  # 샘플 50개만

검색 개선 원리:
  - 법령 조문은 한자어/법률용어가 많아 임베딩 유사도가 낮을 수 있음
  - LLM이 각 청크에 "쉬운 말 요약"과 "검색 키워드"를 부여하면
    구어체 질문과의 의미 거리가 줄어들 수 있음
  - 태깅된 텍스트로 별도 임베딩 인덱스를 빌드하여 성능 비교
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..llm.base import BaseLLMClient, get_llm_client
from .strategies import Chunk

SYSTEM_PROMPT = """\
당신은 법령 및 판례 텍스트를 분석하는 전문가입니다.
주어진 법령 조문 또는 판례 요지를 읽고, 일반인이 검색할 때 쓸 법한
키워드와 쉬운 말 요약을 생성합니다.

[출력 형식 — 반드시 JSON]
{
  "summary": "중학생도 이해할 수 있는 1~2문장 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
}

[규칙]
- summary: 법률 용어를 최대한 쉽게 풀어서 설명. 100자 이내.
- keywords: 일반인이 이 조문/판례를 찾을 때 쓸 법한 검색어. 5개 내외.
  예) "전세금 못 받음", "집주인이 보증금을 돌려주지 않음", "임대차보증금반환"
- 법적 결론을 단언하지 말 것.
"""


class MetadataTagger:
    """청크에 LLM 생성 메타데이터(요약, 키워드)를 추가."""

    def __init__(self, llm: BaseLLMClient | None = None, sleep_between: float = 0.3):
        self._llm = llm
        self.sleep_between = sleep_between  # rate limit 방지

    @property
    def llm(self) -> BaseLLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def tag(self, chunk: Chunk) -> dict:
        """단일 청크에 대한 메타데이터 생성. {summary, keywords} 반환."""
        user_msg = f"다음 텍스트를 분석하세요:\n\n{chunk.text[:800]}"
        try:
            resp = self.llm.complete(
                messages=[{"role": "user", "content": user_msg}],
                system=SYSTEM_PROMPT,
                max_tokens=300,
            )
            text = (resp.text or "").strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            return {
                "summary": data.get("summary", ""),
                "keywords": data.get("keywords", []),
            }
        except Exception:
            return {"summary": "", "keywords": []}

    def tag_chunks(
        self, chunks: list[Chunk], max_chunks: int | None = None
    ) -> list[dict]:
        """청크 목록에 대한 메타데이터 배치 생성."""
        targets = chunks[:max_chunks] if max_chunks else chunks
        results = []
        for i, chunk in enumerate(targets):
            meta = self.tag(chunk)
            results.append({"chunk_id": chunk.chunk_id, **meta})
            if i > 0 and i % 10 == 0:
                print(f"  태깅 진행: {i}/{len(targets)}")
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)
        return results


def enrich_chunk_text(chunk: Chunk, tag: dict) -> Chunk:
    """태그 정보를 청크 텍스트에 추가한 새 Chunk 반환."""
    summary  = tag.get("summary", "")
    keywords = tag.get("keywords", [])
    if not summary and not keywords:
        return chunk

    extra = ""
    if summary:
        extra += f"\n[요약] {summary}"
    if keywords:
        extra += f"\n[키워드] {' '.join(keywords)}"

    return Chunk(
        chunk_id    = chunk.chunk_id + "_tagged",
        doc_type    = chunk.doc_type,
        source_id   = chunk.source_id,
        source_name = chunk.source_name,
        text        = chunk.text + extra,
        metadata    = {**chunk.metadata, "tagged": True,
                       "summary": summary, "keywords": keywords},
    )
