# SASRec Sanity Check Results

## 결과 출처

- 본 문서의 해석은 [sasrec_bpi2012_colab_sanity_check_260511.ipynb](C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/sasrec_bpi2012_colab_sanity_check_260511.ipynb) 실행 결과를 바탕으로 정리했다.
- 결과 확인 및 해석 정리 날짜: `2026-05-11`
- sanity check는 아래 두 model selection 기준에 대해 각각 수행했다.
  - `full_valid_ndcg@10`
  - `full_valid_ndcg@5`

## 목적

- SASRec tuning 결과가 seed 변화에도 재현 가능한지 확인한다.
- validation 기준으로 고른 후보가 test에서도 일관되게 좋은지 확인한다.
- full ranking과 sampled evaluation에서 winner 경향이 크게 뒤집히지 않는지 확인한다.

## 비교 대상

- `anchor_v3_bpi_short_context`
- `refine_v3_ml50_do025`

위 두 후보를 대상으로, `NDCG@10` 기준 모델 선택과 `NDCG@5` 기준 모델 선택을 각각 sanity check 했다.

## 1. Reproducibility

### NDCG@10 기준

- `anchor_v3_bpi_short_context`
  - `full valid NDCG@10 = 0.4859 ± 0.0040`
  - `full test NDCG@10 = 0.4955 ± 0.0405`
- `refine_v3_ml50_do025`
  - `full valid NDCG@10 = 0.5159 ± 0.0176`
  - `full test NDCG@10 = 0.5272 ± 0.0591`

해석:

- anchor는 validation 성능 변동은 작지만, test 변동은 꽤 있는 편이다.
- refine는 seed 간 편차가 전혀 작다고 할 수는 없지만, 평균 성능이 더 높다.
- overall 성능 기준에서는 refine가 더 우세하다.

### NDCG@5 기준

- `anchor_v3_bpi_short_context`
  - `full valid NDCG@5 = 0.3723 ± 0.0498`
  - `full test NDCG@5 = 0.4071 ± 0.1081`
- `refine_v3_ml50_do025`
  - `full valid NDCG@5 = 0.3956 ± 0.0244`
  - `full test NDCG@5 = 0.4229 ± 0.0885`

해석:

- `NDCG@5` 기준에서는 anchor가 seed에 더 민감하게 흔들린다.
- refine는 평균이 더 높고 표준편차도 더 작아서 재현성 측면에서 더 안정적이다.

## 2. Validation / Test 경향

### NDCG@10 기준

- `refine_v3_ml50_do025`는 valid에서 더 높고, test에서도 평균적으로 더 높다.
- 즉, validation 기준으로 고른 후보가 test에서도 대체로 좋은 방향을 유지했다.
- seed에 따라 단일 run의 test 성능이 출렁이는 경우는 있었지만, 평균적으로는 valid-test 경향이 크게 뒤집히지 않았다.

### NDCG@5 기준

- `refine_v3_ml50_do025`는 valid `NDCG@5`와 test `NDCG@5` 모두 anchor보다 평균이 높다.
- `NDCG@5` 기준에서도 validation으로 선택한 방향이 test에서 유지된다고 볼 수 있다.

## 3. Full Ranking / Sampled Evaluation 경향

### NDCG@10 기준

- full ranking에서는 refine가 평균적으로 더 좋은 성능을 보였다.
- sampled evaluation에서도 refine의 `test sampled NDCG@10` 평균이 anchor보다 높았다.
- 즉, 평가 프로토콜이 달라져도 winner가 크게 바뀌지 않았다.

### NDCG@5 기준

- full ranking과 sampled evaluation 모두에서 refine가 전반적으로 더 좋은 평균 성능을 보였다.
- anchor는 일부 seed에서 강하게 나오는 경우가 있었지만, sampled 쪽 변동성도 더 큰 편이었다.

## 4. 종합 해석

- `anchor_v3_bpi_short_context`는 특정 seed에서 test가 높게 나오는 경우가 있었지만, seed 민감도가 더 크다.
- `refine_v3_ml50_do025`는 seed에 따른 변동이 아예 없는 것은 아니지만,
  - 평균 성능이 더 높고
  - validation과 test 경향이 더 일관되고
  - full / sampled 평가 모두에서 더 안정적인 결과를 보였다.

## 최종 결론

- sanity check 결과를 종합하면, 최종 SASRec baseline 후보로는 `refine_v3_ml50_do025`가 더 적절하다.
- 논문에서는 가능하면 단일 best run보다는 다음 형태로 보고하는 것이 바람직하다.
  - `mean ± std` across seeds
  - `full ranking` 기준 main results
  - `sampled evaluation` 기준 supplementary results

## 한 줄 요약

- seed 변화에 대한 재현성, valid/test 일관성, full/sample 평가 일관성을 함께 봤을 때, `refine_v3_ml50_do025`가 `anchor_v3_bpi_short_context`보다 더 안정적이고 설득력 있는 최종 후보로 해석된다.
