"""법률 RAG 에이전트.

질문 유형에 따라 적절한 검색 도구를 선택하고 (tool calling),
검색 결과를 바탕으로 출처가 명시된 답변을 생성한다.

흐름:
  1. 질문 → LLM이 도구 선택 (search_statutes / search_precedents / search_combined)
  2. 도구 실행 → 청크 수집
  3. 청크 + 질문 → LLM 최종 답변 생성 (출처 필수, 디스클레이머 필수)
  4. citation 누락 시 자동 재시도 (최대 1회)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..llm.base import BaseLLMClient, ToolDefinition, get_llm_client
from ..rag.pipeline import LEGAL_DISCLAIMER, SYSTEM_PROMPT, RAGResult, _format_context, _extract_sources
from .tools.statute_search import StatuteSearchTool, SearchResult
from .tools.precedent_search import PrecedentSearchTool, CombinedSearchTool

ROUTER_SYSTEM = """\
당신은 법률 질문을 분류하는 라우터입니다.
주어진 질문을 읽고 가장 적합한 검색 도구를 하나만 선택하세요.

도구 선택 기준:
- search_statutes: 특정 법령 조항·절차·요건을 묻는 경우 ("몇 조에", "법적 요건이", "얼마나 기간이")
- search_precedents: 비슷한 사례의 판결 결과를 묻는 경우 ("판례가", "법원에서", "이런 경우 이겼나요")
- search_combined: 일반 상담형 질문 — 법령 근거와 사례 모두 필요 ("어떻게 해야 하나요", "권리가 있나요", "해결 방법이")

반드시 하나의 도구만 선택하고 적절한 검색 쿼리를 작성하세요.
"""

ANSWER_SYSTEM = SYSTEM_PROMPT  # RAG 파이프라인과 동일한 시스템 프롬프트 재사용


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_used: str
    sources: list[dict] = field(default_factory=list)
    tool_query: str = ""


class LegalAgent:
    """도구 선택(tool calling) 기반 법률 RAG 에이전트."""

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        top_k: int = 5,
        max_tokens: int = 1500,
    ):
        self.llm = llm or get_llm_client()
        self.top_k = top_k
        self.max_tokens = max_tokens

        # 검색 도구 (공유 retriever로 모델 1회만 로드)
        self._statute_tool  = StatuteSearchTool()
        self._prec_tool     = PrecedentSearchTool(self._statute_tool._retriever if self._statute_tool._retriever else None)
        self._combined_tool = CombinedSearchTool()

        self._tool_defs = [
            ToolDefinition(
                name=StatuteSearchTool.name,
                description=StatuteSearchTool.description,
                input_schema=StatuteSearchTool.input_schema,
            ),
            ToolDefinition(
                name=PrecedentSearchTool.name,
                description=PrecedentSearchTool.description,
                input_schema=PrecedentSearchTool.input_schema,
            ),
            ToolDefinition(
                name=CombinedSearchTool.name,
                description=CombinedSearchTool.description,
                input_schema=CombinedSearchTool.input_schema,
            ),
        ]

    # ------------------------------------------------------------------ #
    # 도구 실행                                                            #
    # ------------------------------------------------------------------ #

    def _run_tool(self, tool_name: str, tool_input: dict) -> list[SearchResult]:
        """선택된 도구를 실행하고 SearchResult 목록 반환."""
        if tool_name == StatuteSearchTool.name:
            return self._statute_tool(**tool_input)

        if tool_name == PrecedentSearchTool.name:
            return self._prec_tool(**tool_input)

        if tool_name == CombinedSearchTool.name:
            combined = self._combined_tool(**tool_input)
            return combined["statutes"] + combined["precedents"]

        return []

    # ------------------------------------------------------------------ #
    # 답변 생성                                                            #
    # ------------------------------------------------------------------ #

    def _generate_answer(
        self, question: str, results: list[SearchResult]
    ) -> str:
        if not results:
            return f"관련 법령·판례 정보를 찾지 못했습니다.\n\n{LEGAL_DISCLAIMER}"

        chunks_with_scores = [(r.chunk, r.score) for r in results]
        context = _format_context(chunks_with_scores)

        user_msg = (
            f"[검색된 법령·판례 정보]\n\n{context}\n\n"
            f"[질문]\n{question}\n\n"
            "위 검색 결과를 바탕으로 답변하세요. "
            "답변에 반드시 근거 조문 또는 판례를 명시하고, 법률 자문 면책 고지를 포함하세요."
        )

        resp = self.llm.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=ANSWER_SYSTEM,
            max_tokens=self.max_tokens,
        )
        answer = resp.text or "답변을 생성하지 못했습니다."

        # citation 누락 시 1회 재시도
        if "출처" not in answer and "조" not in answer and "판례" not in answer:
            retry_msg = (
                f"{user_msg}\n\n"
                "⚠️ 이전 답변에 출처(근거 조문/판례)가 누락됐습니다. "
                "반드시 근거 조문명과 조번호 또는 판례 사건번호를 명시하세요."
            )
            resp2 = self.llm.complete(
                messages=[{"role": "user", "content": retry_msg}],
                system=ANSWER_SYSTEM,
                max_tokens=self.max_tokens,
            )
            answer = resp2.text or answer

        if "법률 자문" not in answer and "전문가 상담" not in answer:
            answer += f"\n\n{LEGAL_DISCLAIMER}"

        return answer

    # ------------------------------------------------------------------ #
    # 메인 진입점                                                          #
    # ------------------------------------------------------------------ #

    def run(self, question: str) -> AgentResult:
        """질문 → 도구 선택 → 검색 → 답변 생성."""
        # Step 1: 도구 선택 (LLM tool calling)
        router_resp = self.llm.complete(
            messages=[{"role": "user", "content": question}],
            system=ROUTER_SYSTEM,
            tools=self._tool_defs,
            max_tokens=200,
        )

        tool_name  = "search_combined"  # 기본값
        tool_input = {"query": question}

        if router_resp.tool_calls:
            tc = router_resp.tool_calls[0]
            tool_name  = tc["name"]
            tool_input = tc["input"]
        # LLM이 도구를 선택하지 않은 경우 (텍스트만 반환) → 기본값 유지

        # Step 2: 검색 실행
        results = self._run_tool(tool_name, tool_input)

        # Step 3: 답변 생성
        answer = self._generate_answer(question, results)

        chunks_with_scores = [(r.chunk, r.score) for r in results]
        sources = _extract_sources(chunks_with_scores)

        return AgentResult(
            question  = question,
            answer    = answer,
            tool_used = tool_name,
            tool_query= tool_input.get("query", question),
            sources   = sources,
        )

    def run_to_rag_result(self, question: str) -> RAGResult:
        """Streamlit UI 호환용 — AgentResult → RAGResult 변환."""
        agent_result = self.run(question)
        return RAGResult(
            question         = agent_result.question,
            answer           = agent_result.answer,
            sources          = agent_result.sources,
            retrieved_chunks = [],
        )
