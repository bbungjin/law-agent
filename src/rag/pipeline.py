"""베이스라인 RAG 파이프라인.

흐름: 질문 → 임베딩 검색 → 컨텍스트 구성 → LLM → 답변 + 출처 반환

CLAUDE.md 규칙:
- 출처(근거 조문/판례) 없는 답변 경로 금지
- 확정적 법률 결론 단언 금지 ("반드시", "무조건 승소" 등)
- 모든 답변에 "법률 자문 아님" 디스클레이머 포함
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chunking.strategies import Chunk
from ..llm.base import BaseLLMClient, get_llm_client
from ..retrieval.embedding_retriever import EmbeddingRetriever

LEGAL_DISCLAIMER = (
    "※ 이 답변은 법령·판례 정보를 검색하여 제공한 참고 정보입니다. "
    "법적 효력이 있는 자문이 아니며, 실제 법적 판단은 변호사 등 전문가 상담이 필요합니다."
)

SYSTEM_PROMPT = """\
당신은 법령과 판례를 검색하여 일반인의 법률 관련 질문에 답변하는 정보 검색 보조 도구입니다.

[중요 규칙]
1. 반드시 제공된 검색 컨텍스트(법령 조문, 판례)에 근거하여 답변하세요. 근거 없는 답변은 금지입니다.
2. 답변 마지막에 반드시 출처(근거 조문명+조번호 또는 판례 사건번호)를 명시하세요.
3. "무조건", "반드시 승소합니다", "확실합니다" 등 확정적 법률 결론을 단언하지 마세요.
4. "관련 법령에 따르면", "일반적으로는", "판례에서는" 등의 정보 제공 어조를 유지하세요.
5. 검색된 내용이 질문과 무관하거나 불충분하다면, 솔직하게 "관련 정보를 찾지 못했습니다"라고 말하세요.
6. 답변 마지막에 항상 법률 자문 면책 고지를 포함하세요.
"""


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)  # {doc_type, source_name, ref, score}
    retrieved_chunks: list[Chunk] = field(default_factory=list)


def _format_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """검색된 청크를 LLM 컨텍스트 문자열로 포맷."""
    parts = []
    for chunk, score in chunks_with_scores:
        if chunk.doc_type == "statute":
            ref = f"{chunk.source_name} {chunk.metadata.get('article_no', '')}"
            header = f"[법령: {ref}]"
        else:
            case_no = chunk.metadata.get("case_no", "")
            court = chunk.metadata.get("court", "")
            date = chunk.metadata.get("judgment_date", "")
            ref = f"{court} {date} {case_no}"
            header = f"[판례: {chunk.source_name} / {ref}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(chunks_with_scores: list[tuple[Chunk, float]]) -> list[dict]:
    sources = []
    seen = set()
    for chunk, score in chunks_with_scores:
        if chunk.doc_type == "statute":
            ref = f"{chunk.source_name} {chunk.metadata.get('article_no', '')}"
        else:
            case_no = chunk.metadata.get("case_no", "")
            court = chunk.metadata.get("court", "")
            date = chunk.metadata.get("judgment_date", "")
            ref = f"{chunk.source_name} ({court} {date} {case_no})"
        key = (chunk.doc_type, ref)
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_type": chunk.doc_type,
                "source_name": chunk.source_name,
                "ref": ref,
                "score": round(score, 4),
            })
    return sources


class BaselineRAG:
    """단순 임베딩 검색 → LLM 답변 생성 파이프라인."""

    def __init__(
        self,
        retriever: EmbeddingRetriever,
        llm: BaseLLMClient | None = None,
        top_k: int = 5,
        max_tokens: int = 1500,
    ):
        self.retriever = retriever
        self.llm = llm or get_llm_client()
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, question: str) -> RAGResult:
        """질문에 대한 RAG 답변 생성."""
        # 1. 검색
        results = self.retriever.search(question, top_k=self.top_k)

        if not results:
            return RAGResult(
                question=question,
                answer=f"관련 법령·판례 정보를 찾지 못했습니다.\n\n{LEGAL_DISCLAIMER}",
                sources=[],
            )

        # 2. 컨텍스트 구성
        context = _format_context(results)
        sources = _extract_sources(results)

        # 3. LLM 프롬프트
        user_message = (
            f"[검색된 법령·판례 정보]\n\n{context}\n\n"
            f"[질문]\n{question}\n\n"
            "위 검색 결과를 바탕으로 질문에 답변하세요. "
            "답변 마지막에 출처(근거 조문 또는 판례)를 명시하고, 법률 자문 면책 고지를 포함하세요."
        )

        response = self.llm.complete(
            messages=[{"role": "user", "content": user_message}],
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )

        answer_text = response.text or "답변을 생성하지 못했습니다."

        # 디스클레이머가 누락된 경우 강제 추가
        if "법률 자문" not in answer_text and "전문가 상담" not in answer_text:
            answer_text += f"\n\n{LEGAL_DISCLAIMER}"

        return RAGResult(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieved_chunks=[c for c, _ in results],
        )
