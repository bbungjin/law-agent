# CLAUDE.md

이 문서는 Claude Code(또는 Claude)가 이 저장소에서 작업할 때 따라야 할 컨텍스트와 규칙을 정의합니다.

## 프로젝트 한 줄 요약

일반인의 법률 상담형 질문(예: "전세금 못 받으면 어떻게 해요")에 대해, **법령 조문**과 **판례**를 함께 검색하여 근거를 명시한 답변을 제공하는 **법률 RAG 에이전트**를 구축하고, 그 과정에서 **검색 방식 비교 실험**과 **서빙 최적화 실험**을 수행해 보고서로 정리한다.

포트폴리오 목적: LLM/AI 서비스 개발 직무 지원. 핵심 어필 포인트는 (1) LLM·RAG·에이전트를 처음부터 설계/구현한 경험, (2) 결과가 기대와 다를 때 원인을 분석하고 개선한 사고 과정, (3) 프로파일링 기반 서빙 최적화 경험, (4) 도메인 특화(법률) 데이터 처리 및 신뢰성 설계 경험.

### 도메인 선택 배경 및 범위

- 다루는 문서: 법령(법률·시행령·시행규칙 조문) + 판례(법원 판결문) 둘 다
- 사용자 시나리오: 법률 전문가가 아닌 일반인의 상담형 자연어 질문. 정확한 법조문 citation을 묻는 질문이 아니라 "이런 상황인데 어떻게 해야 하나요" 형태
- 데이터 소스: 국가법령정보 공동활용 Open API (open.law.go.kr) — 법령(`target=law`)과 판례(`target=prec`)를 동일한 API 체계로 제공. 가입 후 API 키 신청 시 보통 1~2일 내 승인. 개발계정 기준 일 10,000건 트래픽 허용
- 데이터 범위(중요, 스코프 제한): 전체 법령/판례를 다루지 않는다. 일반인 상담 수요가 높은 도메인으로 한정 — 예: 주택임대차보호법, 민법(계약·채권 일부), 근로기준법, 소비자기본법 등 10~20개 핵심 법령 + 관련 키워드로 검색한 판례 수백~수천 건. 평가셋 구성과 1개월 일정에 맞추기 위함
- **중요한 제약 — 법적 책임 설계**: 이 시스템은 법률 자문이 아닌 "법령·판례 정보 검색 보조 도구"임을 시스템 프롬프트와 UI 양쪽에 명시한다. 모든 답변에는 (a) 근거 조문/판례 출처를 표시하고 (b) "실제 법적 판단은 변호사 등 전문가 상담이 필요하다"는 안내를 포함한다. 확정적 법률 결론("당신은 무조건 승소합니다" 등)을 단언하는 답변 패턴은 지양하고, 프롬프트 설계 시 이를 명시적으로 제어한다.

## 기간 및 범위

- 총 기간: 약 4주 (1개월 내외)
- 환경: GPU 없음, CPU 전용. 임베딩/리랭커는 경량 사전학습 모델 사용 (파인튜닝은 선택 사항, 시간 남으면 시도)
- LLM: 기본은 API 기반(OpenAI 또는 Anthropic, 환경변수로 키 관리). 로컬 모델(Ollama 등)로 교체 가능하도록 인터페이스 추상화
- 인터페이스: Streamlit (빠른 데모 + 결과 시각화에 적합)
- 데이터: 국가법령정보 공동활용 Open API (open.law.go.kr). 회원가입 → API 활용 신청(보통 1~2일 내 승인) → API Key 발급 → 법령/판례 XML 수집

과도하게 범위를 넓히지 않는다. "전부 다 잘하는 시스템"보다 "각 단계에서 무엇을 시도했고, 왜 그렇게 했고, 결과가 어땠는지"를 명확히 보여주는 것이 목표다. 모든 실험은 before/after 숫자와 함께 기록한다. 법령/판례 전체를 다루지 않고 선정한 핵심 도메인(주택임대차, 근로, 소비자 등)으로 의도적으로 스코프를 제한한다.

## 전체 프로세스 (주차별)

### 1주차 — 데이터 수집 + 파싱 파이프라인 + 베이스라인 RAG

- [ ] open.law.go.kr API 키 신청 (승인까지 1~2일 소요되므로 가장 먼저 처리)
- [ ] 대상 법령 목록 확정 (예: 주택임대차보호법, 민법 일부, 근로기준법, 소비자기본법 등 10~20개)
- [ ] `target=law`로 법령 조문 수집 → 조(條)/항(項)/호(號) 단위로 구조화하여 저장 (XML → JSON 변환)
- [ ] `target=prec`로 관련 키워드 기반 판례 수집 (판례일련번호, 사건명, 사건번호, 선고일자, 법원명, 판시사항/판결요지/판결내용 등 필드 파싱)
- [ ] 법령 조문과 판례를 연결할 수 있는 최소한의 메타데이터 구조 설계 (예: 판례가 인용한 법조항을 텍스트에서 정규식/LLM으로 추출해 링크)
- [ ] 청크 분할 1차 버전: 법령은 "조" 단위를 기본 청크로, 판례는 판시사항/판결요지를 우선 청크로 사용
- [ ] 임베딩 모델 1개로 벡터 인덱스 구축 (예: 한국어 지원 sentence-transformers 모델 — 법률 도메인은 일반 모델 성능이 떨어질 수 있음을 염두에 둘 것)
- [ ] 가장 단순한 형태의 RAG (검색 → LLM 프롬프트 → 답변) 동작 확인
- [ ] 시스템 프롬프트에 "법률 자문 아님" 디스클레이머 및 출처 표시 규칙 포함
- [ ] 평가용 질문-정답 셋(QA set) 20~30개 직접 작성 — 일반인 상담형 질문 + 정답 근거 조문/판례 페어 (수동 평가 기준선)

### 2주차 — 검색 방식 비교 실험 + Feature/Chunk 엔지니어링

- [ ] 청크 전략 변경 실험: 조 단위 vs 조+항 통합 vs LLM 요약 기반 청크 (법령 조문은 짧고 압축적이라 일반 문서와 다른 trade-off가 있을 것으로 예상 — 실제로 어떤지 검증)
- [ ] 검색 방식 비교: BM25(법률 용어는 키워드 매칭이 유리할 수 있음) vs 임베딩 검색(한국어 일반 도메인 모델의 법률 전문용어 처리 한계 확인) vs 하이브리드
- [ ] 일반인 구어체 질문("전세금 못 받으면")과 법률 전문용어("임대차보증금반환") 간의 어휘 격차(lexical gap) 문제를 검색 단계에서 어떻게 완화할지 실험 — 예: LLM으로 질문을 법률 용어로 1차 재작성(query rewriting) 후 검색
- [ ] LLM을 이용한 조문/판례 메타데이터 태깅 (관련 키워드, 쉬운 말 요약 등 부여 후 검색 품질 변화 측정)
- [ ] 위 실험들에서 "기대와 다른 결과"가 나온 부분을 최소 1건 이상 기록하고 원인 분석 (예: query rewriting이 항상 도움이 되지 않았던 사례, 판례 검색에서 하이브리드가 BM25 단독보다 나빴던 이유 등)
- [ ] 정량 지표 정의: Recall@k, MRR 등으로 QA set 기준 비교

### 3주차 — 에이전트화 + 서빙 최적화

- [ ] 질문 유형 분류 라우터 구현 (법령 조문 검색 도구 / 판례 검색 도구 / 두 가지 결합 도구 중 선택, 필요시 질문 재작성 도구도 분리)
- [ ] 각 도구를 함수/tool 형태로 분리, LLM이 tool calling으로 선택하도록 구성
- [ ] 답변 생성 시 반드시 근거 조문/판례를 함께 인용하도록 프롬프트 및 출력 형식 강제 (citation 누락 시 재시도 로직 고려)
- [ ] 임베딩/리랭킹 추론 구간 프로파일링 (Python `cProfile`, `torch.profiler`, 또는 ONNX Runtime 변환 후 비교)
- [ ] 최소 1가지 최적화 적용 (예: 배치 추론, 양자화, ONNX 변환, 캐싱) 후 latency before/after 비교
- [ ] 동시 요청 처리 시 응답 시간 측정 (간단한 부하 테스트)

### 4주차 — UI 통합 + 보고서 정리

- [ ] Streamlit UI에 질의, 답변, 근거 조문/판례 출처(citation) 표시, "법률 자문 아님" 안내 문구, 비교 실험 결과 탭 구성
- [ ] 전체 실험 결과를 `reports/` 디렉토리에 정리 (그래프 + 표 + 분석 텍스트)
- [ ] README 작성: 문제 정의(일반인-법률 정보 간 격차) → 설계 → 실험 → 결과 → 한계 및 회고(법률 도메인 AI의 책임감 있는 설계 관점 포함) 순으로 스토리텔링
- [ ] (시간 여유 시) 작은 임베딩 모델을 법률 QA 페어로 파인튜닝 시도 및 결과 비교

## 디렉토리 구조

```
legal-rag-agent/
├── CLAUDE.md                 # 이 문서
├── README.md                 # 최종 제출용 설명 (4주차에 작성)
├── pyproject.toml            # 또는 requirements.txt
├── .env.example               # API 키 등 환경변수 템플릿
├── data/
│   ├── raw/                  # API로 수집한 원본 XML/JSON (gitignore)
│   ├── processed/            # 조/판례 단위로 구조화된 캐시 (gitignore 대상, 샘플만 커밋)
│   ├── target_laws.yaml      # 수집 대상 법령 목록 (법령명, 법령ID 등)
│   └── qa_set.jsonl          # 평가용 질문-정답 세트 (질문, 정답 근거 조문/판례)
├── src/
│   ├── collection/            # open.law.go.kr API 수집기
│   │   ├── law_api_client.py     # 공통 API 클라이언트 (인증, 페이지네이션)
│   │   ├── statute_collector.py  # 법령(target=law) 수집 + 조 단위 파싱
│   │   └── precedent_collector.py # 판례(target=prec) 수집 + 필드 파싱
│   ├── chunking/              # 청크 전략 (조 단위, 조+항 통합, llm_summary)
│   ├── retrieval/             # BM25, 임베딩, 하이브리드, query rewriting
│   │   ├── bm25_retriever.py
│   │   ├── embedding_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── query_rewriter.py     # 구어체 질문 → 법률 용어 재작성
│   ├── agent/                 # 라우터 + 도구(tool) 정의 + 에이전트 루프
│   │   ├── router.py
│   │   ├── tools/                # statute_search, precedent_search 등
│   │   └── agent.py
│   ├── llm/                   # LLM 클라이언트 추상화 (API/로컬 모델 교체 가능)
│   ├── eval/                  # Recall@k, MRR, latency 측정 스크립트
│   └── serving/                # 프로파일링, 최적화 실험 코드
├── notebooks/                  # 실험용 jupyter/marimo 노트북 (탐색적 분석)
├── reports/                     # 실험 결과 정리 (그래프, 표, 마크다운 분석)
├── app/
│   └── streamlit_app.py        # 최종 데모 UI (디스클레이머 포함)
└── tests/
```

## 코딩 컨벤션

- Python 3.11+, 타입 힌트 적극 사용
- 패키지/의존성 관리는 `pyproject.toml` + `uv` 또는 `pip` 중 하나로 통일 (혼용 금지)
- 모든 실험 스크립트는 결과를 `reports/` 하위에 날짜/실험명 폴더로 저장 (재현 가능하게 시드, 설정값 함께 기록)
- LLM 호출 부분은 반드시 인터페이스로 감싸서 provider 교체가 쉬워야 함 (`src/llm/base.py`의 추상 클래스 상속)
- 커밋 전 큰 바이너리(모델 가중치, 원본 법령/판례 XML 등)는 `.gitignore`에 포함
- BM25 등 키워드 검색을 위한 한국어 토크나이징은 단순 공백 분리 대신 형태소 분석기(`kiwipiepy`)를 사용한다. 영어 위주 라이브러리 기본값을 한국어에 그대로 쓰면 조사/어미 때문에 매칭 품질이 떨어지므로 이 부분을 비교 실험 항목으로도 다룰 것

## 실험 기록 원칙 (중요)

이 프로젝트의 핵심 가치는 "그냥 만들었다"가 아니라 "왜 이렇게 했고, 무엇이 예상과 달랐는가"를 보여주는 것이다. 모든 비교 실험은 다음 템플릿을 따라 `reports/`에 기록한다.

```
## 실험명
- 가설: (무엇을 기대했는가)
- 방법: (무엇을 비교했는가, 데이터/설정)
- 결과: (표/숫자, before-after)
- 분석: (기대와 같았는가/달랐는가, 왜 그런 결과가 나왔다고 생각하는가)
- 다음 액션: (이 결과로 무엇을 바꿨는가)
```

## 하지 말 것

- CTR/CVR, 광고 ML 관련 기능은 이 프로젝트 범위에 포함하지 않음 (별도 프로젝트로 분리하기로 결정됨)
- 과도한 마이크로서비스화, 불필요한 인프라(K8s, 메시지 큐 등) 도입 금지 — 1개월 포트폴리오 프로젝트 규모에 맞게 단순하게 유지
- GPU 필요한 대형 모델 파인튜닝 시도 금지 (CPU 환경 제약)
- vTune은 사용하지 않음 (Python/LLM 서빙 워크로드에 적합하지 않음); 대신 `cProfile`, `py-spy`, `torch.profiler`, 또는 ONNX Runtime 프로파일링 도구 사용
- 전체 법령/판례 데이터베이스를 통째로 수집하지 않음 (스코프 제한된 핵심 도메인만)
- 출처(근거 조문/판례) 없이 답변을 생성하는 경로를 허용하지 않음 — RAG 미사용 일반 지식 답변과 명확히 구분
- 법적 결론을 확정적으로 단언하는 답변 패턴("당신이 승소합니다", "무조건 ~해야 합니다")을 유도하는 프롬프트 작성 금지. 항상 "일반적으로는", "관련 조문에 따르면" 등 정보 제공 어조 유지
- 실제 개인의 민감한 분쟁 정보를 데이터로 수집/저장하지 않음 (공개된 법령·판례 텍스트만 사용)

## 현재 진행 상태

- [x] 0단계: API 키 발급처를 **국가법령정보 공동활용(open.law.go.kr)**로 확정. `.env`에 `LAW_OC=이메일` 설정 필요
- [x] 1주차: 데이터 수집 + 파싱 파이프라인 + 베이스라인 RAG
  - [x] 대상 법령 목록 확정 (`data/target_laws.yaml`)
  - [x] API 클라이언트 (`src/collection/law_api_client.py`) — open.law.go.kr OC 인증, lawSearch/lawService
  - [x] 법령 수집기 (`src/collection/statute_collector.py`) — `<법령><조문단위>` 파싱, 조/항/호 단위
  - [x] 판례 수집기 (`src/collection/precedent_collector.py`) — `<PrecService>` 파싱, 참조조문 포함
  - [x] 청크 전략 1차 버전 (`src/chunking/strategies.py`) — 조 단위 + 판결요지 우선
  - [x] 임베딩 검색기 (`src/retrieval/embedding_retriever.py`) — numpy in-memory 인덱스
  - [x] BM25 검색기 (`src/retrieval/bm25_retriever.py`) — kiwipiepy 형태소 분석 (2주차 비교용)
  - [x] LLM 추상화 구현 (`src/llm/providers/`) — Anthropic / OpenAI provider
  - [x] 베이스라인 RAG 파이프라인 (`src/rag/pipeline.py`) — 디스클레이머 + 출처 강제
  - [x] 평가용 QA 셋 25개 (`data/qa_set.jsonl`)
  - [x] Streamlit UI 뼈대 (`app/streamlit_app.py`)
  - [x] 데이터 수집 완료 (`scripts/collect_data.py`) — 법령 1745조문 / 판례 129건
  - [x] 임베딩 인덱스 빌드 (`scripts/build_index.py`) — 총 2929 청크
  - [x] 베이스라인 평가 (`src/eval/retrieval_eval.py`) — Recall@5: 임베딩 28%, BM25 20%, 하이브리드 24%
- [x] 2주차: 검색 비교 실험 (어휘 격차 완화 포함)
  - [x] 청크 전략 실험: 조 단위 vs 조+항 단위 (`ArticleParagraphStrategy` 추가)
  - [x] 검색 방식 비교: BM25 / 임베딩 / 하이브리드 RRF (`src/retrieval/hybrid_retriever.py`)
  - [x] 어휘 격차 완화: LLM 기반 쿼리 재작성 (`src/retrieval/query_rewriter.py`)
  - [x] 다중 전략 인덱스 빌드 지원 (`scripts/build_index.py --strategy [article|article_paragraph|all]`)
  - [x] 평가 스크립트 확장 (`src/eval/retrieval_eval.py` — --chunk-strategy, --query-rewrite 플래그)
  - [x] 전체 실험 자동화 스크립트 (`scripts/run_week2_experiments.py` — 7가지 조합 비교)
  - [x] 2주차 학습 노트 (`study/week2_study.md`)
  - [x] 실험 실행 및 결과 보고서 생성 (`reports/week2_experiment_results_20260619_2216.md`) — 7가지 조합 완료 (QR 포함)
  - [x] LLM 메타데이터 태깅 구현 및 실험 (`src/chunking/metadata_tagger.py`, `scripts/tag_metadata.py`) — 50개 샘플 태깅, 실험 H 완료
  - [x] 최종 실험 결과 보고서 (`reports/week2_experiment_results_20260619_2230.md`) — 8가지 조합 완료
- [x] 3주차: 에이전트화 + 최적화
  - [x] 질문 유형 분류 라우터 + 도구 구현 (`src/agent/tools/`, `src/agent/agent.py`) — LLM tool calling 기반
  - [x] 법령/판례/통합 검색 도구 분리 (`search_statutes`, `search_precedents`, `search_combined`)
  - [x] citation 누락 시 자동 재시도 로직 구현 (`agent.py` `_generate_answer`)
  - [x] 임베딩 추론 프로파일링 (`src/serving/profiler.py`, `scripts/profile_serving.py`)
  - [x] LRU 캐시 최적화 적용 (`src/serving/cache.py`) — 캐시 히트 시 4974x 속도 향상 (149ms → 0.03ms)
  - [x] 배치 처리량 측정 — CPU 최적 배치 크기: 64 (13.7 q/s)
  - [x] Streamlit UI 업데이트 — 에이전트 모드 탭 + 실험 결과 탭 추가
  - [x] 동시 요청 처리 측정 (`scripts/load_test.py`) — concurrency=2에서 최고 7 req/s, 그 이상은 CPU 경합으로 역효과
  - [x] 3주차 학습 노트 (`study/week3_study.md`)
- [x] 4주차: UI 통합 + 보고서
  - [x] Streamlit UI 6탭 완성 (에이전트/베이스라인/예시질문/실험결과/서빙최적화/시스템소개) — citation·디스클레이머 포함
  - [x] 전체 실험 종합 보고서 (`reports/final_experiment_report.md`) — 7가지 검색 실험 + 3가지 서빙 최적화 실험
  - [x] README.md 작성 — 문제 정의 → 설계 → 실험 결과 → 핵심 발견 → 한계 및 회고
  - [x] 검색 실험 결과 탭에 `final_experiment_report.md` 우선 표시

(작업 진행 시 이 체크리스트를 갱신할 것)
