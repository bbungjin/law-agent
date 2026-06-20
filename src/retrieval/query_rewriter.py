"""LLM 기반 쿼리 재작성 (Query Rewriting).

목적: 일반인 구어체 질문 → 법률 전문용어 쿼리 변환으로 어휘 격차 완화.

예시:
  입력:  "전세금을 못 받았어요. 어떻게 해요?"
  출력:  "임대차보증금반환청구권 행사 절차 및 요건"

사용법:
    from src.retrieval.query_rewriter import QueryRewriter
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("전세금을 못 받았어요")
"""

from __future__ import annotations

import json
from functools import lru_cache

from ..llm.base import BaseLLMClient, get_llm_client

SYSTEM_PROMPT = """\
당신은 법률 분야 전문 용어 변환 보조 도구입니다.
일반인의 구어체 법률 질문을 입력받아, 법령/판례 데이터베이스 검색에 최적화된
법률 전문용어 쿼리로 변환합니다.

[규칙]
1. 원래 질문의 핵심 의미를 유지하면서 법률 전문용어로 바꾸세요.
2. 법령명, 법적 개념, 청구권명 등을 포함하세요.
3. 검색 쿼리이므로 문장이 아닌 키워드/구문 형태로 출력하세요.
4. 반드시 JSON 형식으로 출력하세요: {"rewritten": "변환된 쿼리", "keywords": ["키워드1", "키워드2"]}
5. 변환이 불가능하거나 이미 충분히 법률적이면 원문을 그대로 반환하세요.
"""


class QueryRewriter:
    """LLM을 이용해 구어체 질문을 법률 용어 쿼리로 변환."""

    def __init__(self, llm: BaseLLMClient | None = None):
        self._llm = llm  # 지연 로딩 (필요 시 초기화)

    @property
    def llm(self) -> BaseLLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def rewrite(self, question: str) -> tuple[str, list[str]]:
        """구어체 질문을 법률 용어 쿼리로 변환.

        Returns:
            (rewritten_query, keywords) 튜플.
            실패 시 (원래 질문, []) 반환.
        """
        user_msg = f"다음 질문을 법률 전문용어 검색 쿼리로 변환하세요:\n\n{question}"
        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_msg}],
                system=SYSTEM_PROMPT,
                max_tokens=200,
            )
            text = (response.text or "").strip()
            # JSON 블록 추출
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            rewritten = data.get("rewritten", question)
            keywords = data.get("keywords", [])
            return rewritten, keywords
        except Exception:
            return question, []

    def rewrite_batch(self, questions: list[str]) -> list[tuple[str, list[str]]]:
        """여러 질문 일괄 변환."""
        return [self.rewrite(q) for q in questions]


# 캐시 버전 (동일 질문 반복 시 LLM 호출 생략)
class CachedQueryRewriter(QueryRewriter):
    def __init__(self, llm: BaseLLMClient | None = None):
        super().__init__(llm)
        self._cache: dict[str, tuple[str, list[str]]] = {}

    def rewrite(self, question: str) -> tuple[str, list[str]]:
        if question not in self._cache:
            self._cache[question] = super().rewrite(question)
        return self._cache[question]
