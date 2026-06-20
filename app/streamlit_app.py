"""법률 RAG 에이전트 데모 UI (Streamlit).

실행:
    streamlit run app/streamlit_app.py

전제: scripts/collect_data.py → scripts/build_index.py 순서로 먼저 실행.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.llm.base import get_llm_client
from src.rag.pipeline import BaselineRAG
from src.retrieval.embedding_retriever import EmbeddingRetriever

INDEX_PATH    = ROOT / "data" / "index" / "embedding_index.npy"
PROCESSED_DIR = ROOT / "data" / "processed"
QA_SET_PATH   = ROOT / "data" / "qa_set.jsonl"
REPORTS_DIR   = ROOT / "reports"

DISCLAIMER = (
    "⚠️ 이 서비스는 **법률 자문이 아닙니다**. "
    "법령·판례 정보를 검색하여 참고 정보를 제공하는 도구입니다. "
    "실제 법적 판단은 반드시 변호사 등 전문가와 상담하세요."
)


@st.cache_resource(show_spinner="인덱스 로드 중...")
def load_baseline_rag() -> BaselineRAG | None:
    if not INDEX_PATH.exists():
        return None
    retriever = EmbeddingRetriever()
    retriever.load_index(INDEX_PATH)
    return BaselineRAG(retriever=retriever, llm=get_llm_client(), top_k=5)


@st.cache_resource(show_spinner="에이전트 초기화 중...")
def load_agent():
    try:
        from src.agent.agent import LegalAgent
        return LegalAgent(top_k=5)
    except Exception as e:
        return None


def load_qa_examples() -> list[dict]:
    if not QA_SET_PATH.exists():
        return []
    with open(QA_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_latest_report() -> str:
    final = REPORTS_DIR / "final_experiment_report.md"
    if final.exists():
        return final.read_text(encoding="utf-8")
    reports = sorted(REPORTS_DIR.glob("week2_experiment_results_*.md"), reverse=True)
    if not reports:
        return ""
    return reports[0].read_text(encoding="utf-8")


def load_serving_report() -> str:
    parts = []
    for pattern in ["serving_profile_*.md", "load_test_*.md"]:
        files = sorted(REPORTS_DIR.glob(pattern), reverse=True)
        if files:
            parts.append(files[0].read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.subheader("근거 출처")
    for src in sources:
        icon  = "📖" if src["doc_type"] == "statute" else "⚖️"
        label = "법령" if src["doc_type"] == "statute" else "판례"
        st.markdown(f"{icon} **{label}**: {src['ref']} (유사도: {src['score']:.3f})")


def main() -> None:
    st.set_page_config(page_title="법률 RAG 에이전트", page_icon="⚖️", layout="wide")

    st.title("⚖️ 법령·판례 검색 보조 도구")
    st.warning(DISCLAIMER)

    tab_agent, tab_baseline, tab_examples, tab_experiments, tab_serving, tab_about = st.tabs([
        "에이전트 모드", "베이스라인 RAG", "예시 질문", "검색 실험 결과", "서빙 최적화", "시스템 소개"
    ])

    # ------------------------------------------------------------------ #
    # 탭 1: 에이전트 모드                                                  #
    # ------------------------------------------------------------------ #
    with tab_agent:
        st.markdown("""
        **에이전트 모드**: LLM이 질문 유형을 분석하여 법령 검색 / 판례 검색 / 통합 검색 도구를 자동 선택합니다.
        """)

        agent = load_agent()
        if agent is None:
            st.error("에이전트 초기화 실패. 인덱스 파일을 먼저 빌드하세요.")
        else:
            question = st.text_area(
                "법률 관련 질문을 입력하세요",
                placeholder="예: 전세금을 돌려받지 못하고 있어요. 어떻게 해야 하나요?",
                height=100,
                key="agent_question",
            )

            if st.button("답변 생성 (에이전트)", type="primary") and question.strip():
                with st.spinner("도구 선택 및 검색 중..."):
                    result = agent.run(question)

                col_info, col_answer = st.columns([1, 2])
                with col_info:
                    tool_labels = {
                        "search_statutes":  "📖 법령 검색",
                        "search_precedents": "⚖️ 판례 검색",
                        "search_combined":  "🔍 통합 검색",
                    }
                    st.info(f"**선택된 도구**: {tool_labels.get(result.tool_used, result.tool_used)}")
                    st.caption(f"검색 쿼리: {result.tool_query}")

                with col_answer:
                    st.subheader("답변")
                    st.markdown(result.answer)

                render_sources(result.sources)

    # ------------------------------------------------------------------ #
    # 탭 2: 베이스라인 RAG                                                 #
    # ------------------------------------------------------------------ #
    with tab_baseline:
        rag = load_baseline_rag()
        if rag is None:
            st.error("인덱스 파일이 없습니다. `scripts/build_index.py`를 먼저 실행하세요.")
        else:
            question_b = st.text_area(
                "법률 관련 질문을 입력하세요",
                placeholder="예: 부당해고를 당했는데 어떻게 해야 하나요?",
                height=100,
                key="baseline_question",
            )
            top_k = st.slider("검색 결과 수", 3, 10, 5)

            if st.button("답변 생성 (베이스라인)", type="primary") and question_b.strip():
                rag.top_k = top_k
                with st.spinner("검색 및 답변 생성 중..."):
                    result = rag.answer(question_b)

                st.subheader("답변")
                st.markdown(result.answer)
                render_sources(result.sources)

                with st.expander("검색된 청크 원문 보기"):
                    for chunk in result.retrieved_chunks:
                        st.markdown(f"**[{chunk.doc_type}] {chunk.source_name}**")
                        st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
                        st.divider()

    # ------------------------------------------------------------------ #
    # 탭 3: 예시 질문                                                       #
    # ------------------------------------------------------------------ #
    with tab_examples:
        st.subheader("평가용 예시 질문 (QA 셋 25개)")
        examples = load_qa_examples()
        if not examples:
            st.info("qa_set.jsonl 파일이 없습니다.")
        else:
            domain_filter = st.selectbox(
                "도메인 필터",
                ["전체"] + sorted(set(e.get("domain", "") for e in examples)),
            )
            filtered = examples if domain_filter == "전체" else [
                e for e in examples if e.get("domain") == domain_filter
            ]
            for e in filtered:
                with st.expander(f"[{e['id']}] {e['question']}"):
                    st.markdown(f"**도메인**: {e.get('domain', '-')}")
                    st.markdown(f"**근거 조문**: {e.get('source_statute', '-')}")
                    if e.get("source_precedent"):
                        st.markdown(f"**관련 판례**: {e['source_precedent']}")
                    st.markdown(f"**난이도**: {e.get('difficulty', '-')}")

    # ------------------------------------------------------------------ #
    # 탭 4: 검색 실험 결과                                                  #
    # ------------------------------------------------------------------ #
    with tab_experiments:
        st.subheader("전체 실험 결과 보고서")
        report_md = load_latest_report()
        if report_md:
            st.markdown(report_md)
        else:
            st.info("실험 결과 없음. `scripts/run_week2_experiments.py`를 실행하세요.")

    # ------------------------------------------------------------------ #
    # 탭 5: 서빙 최적화                                                    #
    # ------------------------------------------------------------------ #
    with tab_serving:
        st.subheader("3주차 서빙 최적화 실험 결과")
        serving_md = load_serving_report()
        if serving_md:
            st.markdown(serving_md)
        else:
            st.info("프로파일링 결과 없음. `scripts/profile_serving.py`를 실행하세요.")

    # ------------------------------------------------------------------ #
    # 탭 6: 시스템 소개                                                    #
    # ------------------------------------------------------------------ #
    with tab_about:
        st.subheader("시스템 소개")
        st.markdown("""
        ### 법령·판례 정보 검색 보조 도구

        국가법령정보 공동활용 API(open.law.go.kr)에서 수집한 **법령 조문**과 **판례**를
        기반으로 일반인의 법률 관련 질문에 참고 정보를 제공합니다.

        **수집 대상 법령**
        - 주택임대차보호법, 상가건물 임대차보호법
        - 근로기준법
        - 소비자기본법, 전자상거래 등에서의 소비자보호에 관한 법률
        - 민법 (계약·채권 관련 조항)

        **아키텍처 (3주차 기준)**

        ```
        질문
         ↓
        LegalAgent (tool calling)
         ├─ search_statutes    → 법령 조문 임베딩 검색
         ├─ search_precedents  → 판례 임베딩 검색
         └─ search_combined    → 법령 + 판례 통합 검색
         ↓
        LLM (출처 + 디스클레이머 포함 답변)
        ```

        **한계 및 유의사항**
        - 수집된 법령/판례 범위 내에서만 검색됩니다
        - 최신 개정 내용이 반영되지 않을 수 있습니다
        - 이 시스템은 법률 자문이 아닙니다
        """)


if __name__ == "__main__":
    main()
