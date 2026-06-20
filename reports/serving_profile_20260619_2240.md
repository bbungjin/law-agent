# 서빙 최적화 실험 결과

실험일: 2026-06-19 22:40
모델: jhgan/ko-sroberta-multitask (384차원)
환경: CPU only (GPU 없음)

---

## 1. 단일 쿼리 Latency

| 조건 | 평균 | P95 |
|------|------|-----|
| 캐시 없음 (매번 인코딩) | 149.2ms | 232.3ms |
| 캐시 미스 (첫 호출) | 40.4ms | 201.5ms |
| **캐시 히트 (반복 쿼리)** | **0.03ms** | 0.04ms |

**속도 향상**: 4974x (캐시 히트 vs 캐시 없음)

- 가설: 동일 쿼리 캐싱으로 latency를 크게 줄일 수 있을 것이다.
- 분석: 캐시 히트 시 4974배 빠름. FAQ성 반복 질문이 많은 실제 서비스에서 유효한 최적화.
  단, 캐시는 메모리를 사용하므로 max_size 설정이 중요하다 (현재 256개).

---

## 2. 배치 처리 처리량

| 배치 크기 | 처리량 (queries/sec) | 비고 |
|----------|-------------------|------|
| 1 | 3.6 | 
| 8 | 11.4 | 
| 32 | 11.1 | 
| 64 | 13.7 | 최고

- 최적 배치 크기: **64** (13.7 q/s)
- 분석: CPU 환경에서는 배치 크기 증가의 효과가 GPU보다 제한적.
  단일 요청 처리 중심(온라인 서빙)이라면 캐싱이 더 실용적.
  오프라인 대량 처리(인덱스 재빌드)라면 배치 크기 64 권장.

---

## 3. 최적화 결론

1. **LRU 임베딩 캐시 도입** (`src/serving/cache.py`):
   - 반복 쿼리에서 4974x 속도 향상
   - 구현 비용 낮음, 메모리 overhead 최소 (256개 캐시 ≈ 수 MB)
   - 실시간 Streamlit 앱에 즉시 적용 가능

2. **배치 크기 최적화** (인덱스 빌드 시):
   - 64개 단위 배치가 CPU에서 최고 처리량
   - `scripts/build_index.py`의 BATCH_SIZE를 64로 조정 권장

3. **동시 요청 처리**:
   - CPU 단일 프로세스 기준 단일 요청 처리 후 다음 요청 처리
   - 실제 서비스라면 `asyncio` + 임베딩 스레드 분리 또는 ONNX Runtime 변환 검토

---

## 4. cProfile 상세 분석

```
         15920 function calls (14514 primitive calls) in 0.231 seconds

   Ordered by: cumulative time
   List reduced from 351 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.232    0.232 C:\legal-rag-agent\scripts\profile_serving.py:57(<lambda>)
        1    0.000    0.000    0.231    0.231 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\_contextlib.py:120(decorate_context)
        1    0.001    0.001    0.231    0.231 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\sentence_transformers\util\decorators.py:29(wrapper)
        1    0.001    0.001    0.230    0.230 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\sentence_transformers\sentence_transformer\model.py:483(encode)
        1    0.000    0.000    0.185    0.185 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\sentence_transformers\base\model.py:496(forward)
    216/2    0.002    0.000    0.185    0.092 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\module.py:1774(_wrapped_call_impl)
    216/2    0.004    0.000    0.185    0.092 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\module.py:1782(_call_impl)
        1    0.000    0.000    0.184    0.184 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\sentence_transformers\base\modules\transformer.py:1049(forward)
        1    0.000    0.000    0.184    0.184 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\generic.py:974(wrapper)
        1    0.000    0.000    0.184    0.184 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\output_capturing.py:221(wrapper)
        1    0.000    0.000    0.183    0.183 C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\Li
```
