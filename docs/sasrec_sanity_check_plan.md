# SASRec Sanity Check Plan

## 목적

현재 SASRec tuning 결과는 전반적으로 안정적으로 보이므로, 추가적인 대규모 탐색보다는 **간단한 sanity check**만 수행한 뒤 다음 단계로 넘어간다.  
여기서 sanity check는 아래 4가지를 뜻한다.

1. 최종 후보 모델 1~2개를 고른다.
2. 각 후보를 seed 2~3개로 다시 실행해본다.
3. 결과의 평균과 표준편차를 간단히 확인한다.
4. validation 기준 선택이 test에서도 크게 어긋나지 않는지 확인한다.

## 왜 하는가

- 한 번의 best score가 우연히 잘 나온 결과인지 확인하기 위해
- seed가 바뀌어도 비슷한 성능이 나오는지 확인하기 위해
- `valid` 기준으로 고른 모델이 `test`에서도 합리적인 선택인지 확인하기 위해
- 다음 단계 실험으로 넘어가기 전에 baseline이 충분히 신뢰할 만한지 점검하기 위해

## 현재 결과 기준 후보

현재 엑셀 정리 결과를 기준으로 sanity check 대상은 아래 2개로 둔다.

### 후보 1: `refine_v3_ml50_do030`

- `full valid NDCG@10` 기준 최고 후보
- 현재 설정:
  - `maxlen=50`
  - `dropout_rate=0.30`
  - `hidden_units=50`
  - `num_blocks=2`
  - `num_heads=1`
  - `lr=0.001`

### 후보 2: `refine_v3_ml50_do025`

- `full test NDCG@10` 기준으로도 유력한 비교 후보
- 현재 설정:
  - `maxlen=50`
  - `dropout_rate=0.25`
  - `hidden_units=50`
  - `num_blocks=2`
  - `num_heads=1`
  - `lr=0.001`

## 현재까지 확보된 seed

### `refine_v3_ml50_do030`

- `seed=42`
- `seed=2024`

### `refine_v3_ml50_do025`

- `seed=42`
- `seed=2024`

## 추가로 실행할 최소 실험

sanity check를 최소한으로 끝내려면 아래 2개만 추가하면 된다.

- `refine_v3_ml50_do030_seed7`
- `refine_v3_ml50_do025_seed7`

즉, 각 후보당 seed 3개:

- `42`
- `2024`
- `7`

를 확보하면 된다.

## 무엇을 볼 것인가

### 1. 평균

각 후보에 대해 아래 지표의 평균을 본다.

- `full valid NDCG@10`
- `full test NDCG@10`

필요하면 보조로:

- `full valid MRR`
- `full test MRR`

도 함께 본다.

### 2. 표준편차

각 후보에 대해 아래 지표의 표준편차를 본다.

- `full valid NDCG@10`
- `full test NDCG@10`

표준편차가 너무 크면:

- seed에 민감한 설정일 수 있고
- 결과 재현성이 낮을 수 있다.

### 3. validation/test 경향

아래를 확인한다.

- `valid`에서 좋은 후보가 `test`에서도 대체로 좋게 유지되는가
- seed를 바꿔도 두 후보의 상대적인 경향이 완전히 뒤집히지 않는가

예를 들어:

- 어떤 후보가 `valid`에서는 계속 좋지만 `test`에서는 계속 약하다면
  - validation selection의 타당성을 다시 점검해야 한다.

## 기대하는 결론

이 sanity check의 목표는 “최종 best model을 아주 엄밀하게 확정”하는 것보다, 아래를 확인하는 데 있다.

- 현재 선택한 SASRec 설정이 대체로 재현 가능한가
- 선택 기준으로 사용한 `full valid NDCG@10`이 납득 가능한가
- 다음 단계 실험으로 넘어가도 될 만큼 baseline이 안정적인가

## 실행 후 정리 방식

후보별로 아래 형태로 정리하면 충분하다.

### 후보별 기록

- run name
- seed
- `full valid NDCG@10`
- `full test NDCG@10`
- `full valid MRR`
- `full test MRR`

### 최종 요약

- 후보 1 평균 ± 표준편차
- 후보 2 평균 ± 표준편차
- 어느 후보가 더 안정적인지 한 줄 결론

## 한 줄 요약

현재 단계에서는 `refine_v3_ml50_do030`과 `refine_v3_ml50_do025`를 대상으로 `seed=7`만 추가 실행하고,  
세 seed(`42`, `2024`, `7`) 기준의 평균/표준편차와 valid-test 경향만 확인하면 sanity check로 충분하다.
