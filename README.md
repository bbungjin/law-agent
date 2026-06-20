# 법령·판례 검색 보조 도구 (Legal RAG Agent)

> 일반인의 법률 상담형 질문에 법령 조문과 판례를 근거로 답변하는 RAG 에이전트  
> 포트폴리오 프로젝트 (LLM/AI 서비스 개발 직무) — 4주 완성

---

## 문제 정의

일반인이 "전세금을 못 받았는데 어떻게 해야 하나요?"라고 물으면, 관련 법령이 무엇인지, 비슷한 판례가 있는지 알기 어렵다. 법령 원문은 법률 전문용어로 쓰여 있고, 판례는 방대한 양의 문서에 흩어져 있다.

**핵심 어려움 — 어휘 격차(Lexical Gap)**:

```
사용자 질문:  "전세금 못 받았어요"
법령 용어:   "임대차보증금반환청구권"
```

일반인 구어체와 법률 전문용어 사이의 표현 격차가 키워드 검색을 어렵게 만든다.

**이 프로젝트의 목표**: 이 격차를 어떻게 줄일 수 있는지 실험하고, 일반인이 법령·판례 정보에 접근할 수 있는 보조 도구를 구축한다.

> **법적 책임 고지**: 이 시스템은 법률 자문이 아닙니다. 법령·판례 정보를 검색하여 참고 정보를 제공하는 도구입니다. 실제 법적 판단은 반드시 변호사 등 전문가와 상담하세요.

---

## 아키텍처

### 전체 파이프라인

```
[데이터 수집]
  국가법령정보 API (open.law.go.kr)
  ├── 법령 조문 (target=law): 1,745개 조문
  └── 판례 (target=prec): 129건
         ↓
[청크 분할 + 임베딩 인덱스 빌드]
  법령: 조(條) 단위 청크
  판례: 판시사항 + 판결요지 청크
  임베딩 모델: jhgan/ko-sroberta-multitask (384차원)
  총 청크: 2,929개
         ↓
[에이전트 모드 — 3주차]
  사용자 질문
      ↓
  LegalAgent (LLM tool calling)
    ├─ search_statutes    → 법령 조문 임베딩 검색
    ├─ search_precedents  → 판례 임베딩 검색
    └─ search_combined    → 법령 + 판례 통합 검색
      ↓
  LLM (Anthropic Claude / OpenAI)
  → 출처 인용 + 법률 자문 아님 디스클레이머 포함 답변
```

### 스택

| 범주 | 선택 | 이유 |
|------|------|------|
| 임베딩 | jhgan/ko-sroberta-multitask | 한국어 지원 경량 모델, CPU 추론 가능 |
| 키워드 검색 | BM25 + kiwipiepy | 한국어 형태소 분석 필수 (조사/어미 처리) |
| 하이브리드 | RRF (Reciprocal Rank Fusion) | 임베딩 + BM25 점수 결합 |
| LLM | Anthropic Claude / OpenAI (교체 가능) | 인터페이스 추상화로 provider 독립 |
| UI | Streamlit | 빠른 데모 + 실험 결과 시각화 |
| 환경 | CPU only, Python 3.11 | GPU 없이 동작 |

---

## 실험 결과

### 검색 성능 비교 (2주차, Recall@5 기준, QA 25개)

| # | 실험 조합 | Recall@5 | MRR | 베이스라인 대비 |
|---|---------|----------|-----|--------------|
| A | 임베딩 / 조 단위 (베이스라인) | 28.0% | 0.2333 | 기준 |
| B | 임베딩 / 조+항 단위 | 28.0% | 0.1980 | MRR ↓ |
| C | BM25 / 조 단위 | 20.0% | 0.1800 | **-8%p** |
| D | BM25 / 조+항 단위 | 20.0% | 0.1800 | -8%p |
| E | 하이브리드 / 조 단위 | 24.0% | 0.2000 | -4%p |
| F | 임베딩 + Query Rewriting | 24.0% | 0.1867 | **-4%p** ← 역효과 |
| G | 하이브리드 + Query Rewriting | **28.0%** | **0.2500** | MRR 최고 |
| H | 임베딩 + 메타데이터 태깅 | 28.0% | 0.2300 | 태깅 효과 미미 |

### 서빙 최적화 결과 (3주차)

| 최적화 방법 | Before | After | 개선 |
|-----------|--------|-------|------|
| LRU 임베딩 캐시 | 96.6ms/req | **0.02ms/req** | **4,830배** |
| 배치 처리 (size=64) | 8.1 q/s | **21.8 q/s** | 2.7배 |
| 동시 요청 (concurrency=4) | 8.97 req/s | 10.51 req/s | 1.2배 (한계 존재) |

---

## 핵심 발견 (기대와 다른 결과)

### 1. 하이브리드 검색이 임베딩 단독보다 낮았다

> "하이브리드가 항상 더 낫다"는 통념이 소규모 법률 도메인에서는 성립하지 않았다.

RRF가 BM25 노이즈를 임베딩 점수에 희석시켜 Recall이 오히려 낮아졌다 (28% → 24%). 판례 수가 129건밖에 되지 않는 소규모 인덱스에서 BM25의 키워드 매칭 품질이 떨어져 하이브리드가 손해를 봤다.

### 2. Query Rewriting이 검색기에 따라 반대 효과를 낸다

| 검색기 | QR 효과 | 이유 |
|-------|---------|------|
| 임베딩 단독 | -4%p (역효과) | 임베딩 모델이 이미 구어체를 의미론적으로 잘 처리. QR이 과도한 법률 용어화로 방해 |
| 하이브리드 | +4%p, MRR +0.05 (효과적) | QR로 법률 키워드가 명시되면 BM25 성분의 키워드 매칭 향상 → RRF 점수 개선 |

**결론**: QR 적용 여부를 검색기 종류에 따라 선택적으로 제어해야 한다.

### 3. CPU 동시 요청은 선형 스케일이 되지 않는다

ThreadPoolExecutor로 동시 요청을 늘려도 처리량이 선형으로 증가하지 않았다. PyTorch CPU 임베딩이 내부적으로 이미 멀티스레드를 사용하기 때문에 외부 스레드 추가 시 오히려 경합이 발생한다. 동시성=8에서 동시성=4보다 처리량이 낮아졌다.

---

## 실행 방법

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정 (.env 파일 생성)
cp .env.example .env
# .env에 아래 값 입력:
# LAW_OC=your_email@example.com   # open.law.go.kr 가입 이메일
# ANTHROPIC_API_KEY=sk-ant-...    # Anthropic API 키
# LLM_PROVIDER=anthropic           # 또는 openai
```

### 2. 데이터 수집 + 인덱스 빌드

```bash
# 법령·판례 수집 (API 키 필요, 약 10분 소요)
python scripts/collect_data.py

# 임베딩 인덱스 빌드 (약 5분 소요)
python scripts/build_index.py
```

### 3. Streamlit UI 실행

```bash
streamlit run app/streamlit_app.py
```

### 4. 에이전트 CLI 테스트

```bash
# 기본 예시 질문 3개 실행
python scripts/run_agent.py

# 단일 질문
python scripts/run_agent.py --question "전세금을 못 받았어요. 어떻게 해야 하나요?"
```

### 5. 실험 재현

```bash
# 2주차 검색 비교 실험 전체 실행 (8가지 조합)
python scripts/run_week2_experiments.py

# 서빙 최적화 프로파일링
python scripts/profile_serving.py

# 동시 요청 부하 테스트
python scripts/load_test.py
```

---

## 디렉토리 구조

```
legal-rag-agent/
├── app/
│   └── streamlit_app.py        # 6탭 UI (에이전트/베이스라인/예시/실험결과/서빙/소개)
├── data/
│   ├── target_laws.yaml        # 수집 대상 법령 목록
│   ├── qa_set.jsonl            # 평가용 QA 25개
│   ├── processed/              # 구조화된 조문·판례 (gitignore)
│   └── index/                  # 임베딩 인덱스 .npy (gitignore)
├── reports/
│   ├── final_experiment_report.md   # 전체 실험 종합 보고서
│   ├── week1_baseline_retrieval.md  # 1주차 베이스라인
│   ├── week2_experiment_results_*.md # 2주차 검색 비교
│   ├── serving_profile_*.md         # 서빙 최적화
│   └── load_test_*.md               # 부하 테스트
├── src/
│   ├── collection/             # open.law.go.kr API 수집기
│   ├── chunking/               # 청크 전략 + 메타데이터 태깅
│   ├── retrieval/              # BM25 / 임베딩 / 하이브리드 / Query Rewriting
│   ├── agent/                  # LegalAgent + tool 정의
│   ├── llm/                    # LLM 클라이언트 추상화 (provider 교체 가능)
│   ├── eval/                   # Recall@k, MRR 평가 스크립트
│   └── serving/                # LRU 캐시 + 프로파일러
├── scripts/
│   ├── collect_data.py         # 데이터 수집
│   ├── build_index.py          # 임베딩 인덱스 빌드
│   ├── run_week2_experiments.py # 검색 실험 자동화
│   ├── run_agent.py            # 에이전트 CLI
│   ├── profile_serving.py      # 서빙 최적화 실험
│   └── load_test.py            # 부하 테스트
└── study/
    ├── week1_study.md          # RAG 기초 학습 노트
    ├── week2_study.md          # 검색 비교 실험 학습 노트
    └── week3_study.md          # 에이전트화·서빙 최적화 학습 노트
```

---

## 한계 및 회고

### 기술적 한계

| 한계 | 내용 | 개선 방향 |
|------|------|---------|
| 낮은 Recall (28%) | 판례 수 부족 (129건), 평가셋 소규모 (25개) | 판례 추가 수집, 평가셋 확대 |
| 하이브리드 미최적화 | RRF 가중치 기본값 사용 | BM25 가중치 튜닝 실험 |
| 메타데이터 태깅 미완 | 전체 2929 청크 중 100개만 샘플 태깅 | 전체 태깅 후 재측정 |
| 임베딩 모델 | 일반 한국어 모델 사용 | 법률 QA 쌍 파인튜닝 시 성능 향상 기대 |
| CPU 환경 제약 | 동시성 한계, 배치 효과 제한 | GPU 환경에서 ONNX 변환 효과 더 클 것 |

### 설계 관점 회고 — 법률 도메인 AI의 책임

이 프로젝트에서 가장 고민한 부분은 기술이 아니라 **법적 책임 설계**였다.

- 법률 AI가 "당신은 승소합니다"처럼 확정적 결론을 내리면 사용자가 전문가 상담을 대체할 수 있고, 잘못된 결론이 실제 피해로 이어질 수 있다.
- 따라서 모든 답변에 (a) 근거 조문·판례 출처 명시, (b) "법률 자문 아님" 안내를 강제했다.
- 출처 없는 답변 경로를 시스템 수준에서 차단했고 (citation 누락 시 자동 재시도), UI와 시스템 프롬프트 양쪽에서 이를 명시했다.

**기술 성능(Recall 28%)보다 책임 있는 설계가 더 중요한 도메인임을 이 프로젝트를 통해 확인했다.**

### 프로젝트를 통해 배운 것

1. **실험이 기대를 벗어날 때가 더 중요하다**: QR 역효과, 하이브리드 열세 — 이런 결과를 기록하고 원인을 분석하는 과정이 단순히 "높은 숫자"를 만드는 것보다 가치 있었다.
2. **도메인 특화가 일반 방법론보다 중요**: 영어 RAG의 best practice가 한국어 법률 도메인에서는 다르게 작동했다. kiwipiepy 형태소 분석, 법령의 "조" 단위 청크, 구어체-법률용어 어휘 격차 모두 도메인 특화 문제였다.
3. **단순한 최적화가 가장 효과적**: LRU 캐시(코드 50줄) 하나로 4830배 속도 향상. 복잡한 ONNX 변환보다 먼저 단순한 캐싱을 고려해야 했다.

---

## 환경 변수 (.env.example)

```
# 국가법령정보 공동활용 API (open.law.go.kr 가입 이메일)
LAW_OC=your_email@example.com

# LLM Provider (anthropic 또는 openai)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

*이 프로젝트는 포트폴리오 목적으로 제작되었습니다. 국가법령정보 공동활용 API 데이터를 사용합니다.*
