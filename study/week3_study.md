# 3주차 학습 노트: 에이전트화 + 서빙 최적화

---

## 이번 주 목표가 무엇인가?

2주차에서 "어떤 검색이 더 좋은가"를 실험했다면, 3주차는 두 가지를 합니다.

1. **에이전트화**: 단순 RAG → "LLM이 도구를 선택하는" 에이전트로 업그레이드
2. **서빙 최적화**: 실제 서비스에서 쓸 수 있도록 속도 문제를 측정하고 개선

---

## 핵심 개념 1: RAG와 에이전트의 차이

### 기존 베이스라인 RAG (1주차)

```
사용자 질문
    ↓
임베딩 검색 (항상 같은 방식)
    ↓
LLM에게 검색 결과 + 질문 전달
    ↓
답변
```

모든 질문에 대해 항상 같은 검색 방법(임베딩)을 씁니다.

### 3주차 에이전트

```
사용자 질문
    ↓
LLM이 질문을 읽고 도구 선택
  ├─ "법령 조항을 찾아라" → search_statutes
  ├─ "판례를 찾아라" → search_precedents
  └─ "상담형 질문" → search_combined
    ↓
선택된 도구로 검색 실행
    ↓
검색 결과 + 질문 → LLM 답변 (출처 필수)
```

LLM이 "어떤 도구를 쓸지" 스스로 결정합니다.

---

## 핵심 개념 2: Tool Calling (함수 호출)

Tool Calling은 LLM에게 "이런 함수들이 있어. 필요하면 불러"라고 알려주는 기능입니다.

### 코드로 보면

```python
tools = [
    ToolDefinition(
        name="search_statutes",
        description="법령 조문을 검색합니다. 특정 법조항을 찾을 때 사용.",
        input_schema={"query": "string", "top_k": "integer"},
    ),
    ToolDefinition(
        name="search_precedents",
        description="판례를 검색합니다. 비슷한 사례의 판결 결과를 찾을 때.",
        input_schema={"query": "string"},
    ),
]

# LLM 응답에 tool_calls가 포함됨
response = llm.complete(
    messages=[{"role": "user", "content": "전세금 못 받는 판례 있나요?"}],
    tools=tools,
)
# response.tool_calls = [{"name": "search_precedents", "input": {"query": "임대차보증금 반환 판례"}}]
```

LLM이 응답할 때 텍스트 대신 "이 함수를 이 인자로 호출해줘"라는 신호를 보냅니다.

### 이 프로젝트의 도구 3개

| 도구 | 언제 선택 |
|------|-----------|
| `search_statutes` | "법 몇 조에", "요건이", "기간이 얼마나" |
| `search_precedents` | "판례가", "법원에서", "이긴 사례 있나요" |
| `search_combined` | "어떻게 해야 하나요", "권리가 있나요" (일반 상담) |

---

## 핵심 개념 3: Citation 강제 (출처 누락 방지)

법률 답변에서 출처(근거 조문/판례)가 빠지면 안 됩니다. 두 가지 장치를 씁니다.

1. **시스템 프롬프트에 명시**: "답변에 반드시 근거 조문 또는 판례를 인용하세요"
2. **자동 재시도**: 답변에 "조", "판례", "출처" 등이 없으면 LLM에게 다시 요청

```python
if "출처" not in answer and "조" not in answer and "판례" not in answer:
    # 재시도
    resp2 = llm.complete(messages=[{"role": "user", "content": retry_msg}])
```

---

## 핵심 개념 4: 임베딩 캐시 (LRU Cache)

### 왜 필요한가?

임베딩 모델이 쿼리를 처리하는 데 CPU에서 약 **150ms**가 걸립니다.
같은 질문을 두 번 하면 두 번 다 150ms를 씁니다. 낭비입니다.

### LRU 캐시란?

LRU(Least Recently Used) = 가장 오래 안 쓴 것을 먼저 버리는 방식

```
질문 "전세금 못 받아요" → 처음: 150ms (계산)
질문 "전세금 못 받아요" → 두 번째: 0.03ms (캐시에서 꺼냄)  ← 4974배 빠름!
```

```python
cache = LRUEmbeddingCache(max_size=256)  # 최대 256개 쿼리 저장

vec = cache.get(query)   # 캐시에 있으면 즉시 반환
if vec is None:
    vec = model.encode(query)   # 없으면 계산
    cache.put(query, vec)       # 캐시에 저장
```

### 실제 측정 결과

| 조건 | 응답시간 | 비교 |
|------|---------|------|
| 캐시 없음 | 149ms | 기준 |
| 캐시 히트 | 0.03ms | **4974배 빠름** |

FAQ성 질문이 반복되는 실제 서비스에서 매우 효과적입니다.

---

## 핵심 개념 5: 배치 처리

여러 텍스트를 한 번에 임베딩하면 하나씩 처리하는 것보다 효율적입니다.

```
batch=1 (하나씩):  3.6 queries/sec
batch=64 (한꺼번에): 13.7 queries/sec  ← 3.8배 빠름
```

인덱스를 새로 빌드할 때 (2929개 청크) `BATCH_SIZE=64`로 설정하면 더 빠르게 완료됩니다.
(온라인 서빙에서는 캐시가 더 실용적)

---

## 핵심 개념 6: 동시 요청 처리

실제 서비스에서 여러 사용자가 동시에 질문하면 어떻게 되나?

```
동시성=1:  ~150ms/request
동시성=4:  응답시간 증가 (CPU 경합)
```

CPU 환경에서 임베딩은 PyTorch가 내부적으로 멀티스레드를 사용하므로,
외부에서 더 많은 스레드를 추가해도 오히려 느려질 수 있습니다.

해결책:
- **단기**: LRU 캐시로 반복 요청 즉시 처리
- **중기**: FastAPI + asyncio로 LLM API 호출을 비동기 처리
- **장기**: ONNX Runtime 변환으로 임베딩 추론 최적화

---

## 이번 주 구현된 것들

| 파일 | 역할 |
|------|------|
| `src/agent/tools/statute_search.py` | 법령 검색 도구 정의 |
| `src/agent/tools/precedent_search.py` | 판례/통합 검색 도구 정의 |
| `src/agent/agent.py` | LegalAgent — tool calling 에이전트 |
| `src/serving/cache.py` | LRU 임베딩 캐시 |
| `src/serving/profiler.py` | 성능 측정 유틸 |
| `scripts/run_agent.py` | 에이전트 CLI 테스트 |
| `scripts/profile_serving.py` | 서빙 최적화 실험 |
| `scripts/load_test.py` | 동시 요청 부하 테스트 |
| `app/streamlit_app.py` | 에이전트 모드 탭 추가 |

---

## 실행 순서

```bash
# 1. 에이전트 동작 확인 (LLM API 필요)
python scripts/run_agent.py

# 2. 서빙 프로파일링
python scripts/profile_serving.py

# 3. 부하 테스트
python scripts/load_test.py

# 4. Streamlit UI 실행
streamlit run app/streamlit_app.py
```
