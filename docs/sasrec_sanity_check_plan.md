# SASRec Sanity Check Plan

## 목적

현재 SASRec tuning 결과는 전반적으로 안정적으로 보이므로, 추가적인 대규모 탐색보다는 **간단한 sanity check**만 수행한 뒤 다음 단계로 넘어간다.

여기서 sanity check의 목적은 아래와 같다.

1. 현재 선택한 후보 모델이 seed를 바꿔도 비슷한 성능을 내는지 확인한다.
2. 모델 선택 기준으로 사용한 validation metric이 test에서도 납득 가능한 방향으로 이어지는지 확인한다.
3. 최종적으로 baseline SASRec 결과가 reproducibility 측면에서 충분히 안정적인지 점검한다.

## 기본 원칙

- 모델 선택은 항상 `valid` 기준으로 한다.
- 최종 보고는 `test` 성능으로 한다.
- sanity check에서는 **선택 기준(metric)** 별로 따로 확인한다.
- 즉, `NDCG@10` 기준 sanity check와 `NDCG@5` 기준 sanity check를 분리해서 본다.

## 현재 실험 구조

현재까지 실험은 두 종류의 model selection 기준으로 정리되어 있다.

1. `full_valid_ndcg@10` 기준으로 best epoch를 선택한 run들
2. `full_valid_ndcg@5` 기준으로 best epoch를 선택한 run들

또한 각 run에 대해 아래 지표들을 함께 보고 있다.

- `full valid/test NDCG@10`
- `full valid/test NDCG@5`
- `full valid/test HR@10`
- `full valid/test HR@5`
- `full valid/test MRR`
- `sampled valid/test NDCG@10`
- `sampled valid/test NDCG@5`
- `sampled valid/test HR@10`
- `sampled valid/test HR@5`
- `sampled valid/test MRR`

## Sanity Check 방식

sanity check는 아래 순서로 진행한다.

1. selection metric 기준으로 최종 후보 1~2개를 고른다.
2. 각 후보를 seed 2~3개로 다시 실행한다.
3. selection metric에 해당하는 valid/test 성능의 평균과 표준편차를 계산한다.
4. 같은 run들에서 다른 보조 지표들도 함께 확인한다.
5. valid와 test의 경향이 크게 뒤집히지 않는지 확인한다.

## A. `NDCG@10` 기준 sanity check

### 현재 기준

- model selection 기준: `full_valid_ndcg@10`

### 후보

현재 결과를 기준으로 아래 2개를 sanity check 대상으로 둔다.

- `anchor_v3_bpi_short_context_seed42`
- `refine_v3_ml50_do025_seed2024`

위 2개는 사용자가 최종 후보로 보고 있는 설정이므로, `NDCG@10` 기준에서도 이 두 설정의 안정성을 확인한다.

### 추가 실행

각 후보에 대해 seed를 2~3개 맞춘다.

예:

- `seed=42`
- `seed=2024`
- `seed=7`

즉, 각 설정에 대해 아직 없는 seed만 추가 실행하면 된다.

### 중심 확인 지표

`NDCG@10` 기준 sanity check에서는 아래를 중심으로 본다.

- `full valid NDCG@10`
- `full test NDCG@10`

### 보조 확인 지표

같은 run에서 아래 지표도 함께 확인한다.

- `full valid/test NDCG@5`
- `full valid/test MRR`
- `sampled valid/test NDCG@10`
- `sampled valid/test NDCG@5`
- `sampled valid/test MRR`

## B. `NDCG@5` 기준 sanity check

### 현재 기준

- model selection 기준: `full_valid_ndcg@5`

### 후보

`NDCG@5` 기준 정리 결과를 바탕으로 아래 후보들을 sanity check 대상으로 둔다.

- `anchor_v3_bpi_short_context_seed42`
- `refine_v3_ml50_do025_seed2024`

즉, 사용자가 최종적으로 보고 있는 동일한 두 후보를 `NDCG@5` 기준에서도 다시 점검한다.

### 추가 실행

마찬가지로 각 후보에 대해 seed를 2~3개 맞춘다.

예:

- `seed=42`
- `seed=2024`
- `seed=7`

### 중심 확인 지표

`NDCG@5` 기준 sanity check에서는 아래를 중심으로 본다.

- `full valid NDCG@5`
- `full test NDCG@5`

### 보조 확인 지표

같은 run에서 아래 지표도 함께 확인한다.

- `full valid/test NDCG@10`
- `full valid/test MRR`
- `sampled valid/test NDCG@10`
- `sampled valid/test NDCG@5`
- `sampled valid/test MRR`

## 평균과 표준편차를 왜 보는가

각 후보를 여러 seed로 실행한 뒤, 아래를 계산한다.

- 평균
- 표준편차

의미는 다음과 같다.

- 평균: 이 설정이 전반적으로 어느 정도 성능을 내는지 보기 위함
- 표준편차: seed가 바뀌었을 때 결과가 얼마나 흔들리는지 보기 위함

즉:

- 평균이 높고
- 표준편차가 작으면

그 설정은 **성능도 좋고 안정적**이라고 해석할 수 있다.

## valid / test 경향을 왜 보는가

sanity check에서는 단순히 평균만 보는 것이 아니라,

- `valid`에서 좋게 선택된 후보가
- `test`에서도 대체로 좋게 이어지는지

를 함께 확인해야 한다.

확인할 내용:

- valid 기준 winner가 test에서도 크게 무너지지 않는가
- seed를 바꿔도 valid와 test의 상대적 경향이 완전히 뒤집히지 않는가
- full과 sampled 결과가 완전히 모순되지 않는가

좋은 경우:

- valid 평균이 높은 후보가 test 평균도 높음
- seed를 바꿔도 경향이 비슷함
- 표준편차가 크지 않음

조심해야 하는 경우:

- valid에서는 계속 좋은데 test에서는 계속 약함
- seed마다 winner가 심하게 바뀜
- full과 sampled가 전혀 다른 결론을 줌

## 최종 정리 방식

각 selection 기준별로 아래 형태의 표를 만들면 충분하다.

### 후보별 기록

- run name
- seed
- `full valid NDCG@10`
- `full test NDCG@10`
- `full valid NDCG@5`
- `full test NDCG@5`
- `full valid MRR`
- `full test MRR`
- `sampled valid NDCG@10`
- `sampled test NDCG@10`
- `sampled valid NDCG@5`
- `sampled test NDCG@5`
- `sampled valid MRR`
- `sampled test MRR`

### 기준별 요약

- `NDCG@10` 기준:
  - 후보 1 평균 ± 표준편차
  - 후보 2 평균 ± 표준편차
  - valid/test 경향 한 줄 요약

- `NDCG@5` 기준:
  - 후보 1 평균 ± 표준편차
  - 후보 2 평균 ± 표준편차
  - valid/test 경향 한 줄 요약

## 한 줄 요약

이번 sanity check는 `NDCG@10` 기준과 `NDCG@5` 기준을 **각각 따로** 보고,  
선택한 두 후보(`anchor_v3_bpi_short_context_seed42`, `refine_v3_ml50_do025_seed2024`)를 여러 seed로 다시 실행한 뒤,  
평균/표준편차와 valid-test 경향을 함께 확인하는 방식으로 진행한다.
