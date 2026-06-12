# 보고서 작성용 컨텍스트 정리

이 문서는 최종 엑셀 파일 외에, 보고서 작성 시 ChatGPT가 알아야 할 실험 맥락을 정리한 것이다.

## 1. 연구 단계 구조

본 연구는 아래 3단계로 진행되었다.

1. Stage 1
- SASRec 기반 single-task baseline
- 입력: item sequence
- 출력: next activity prediction

2. Stage 2
- Stage 1 baseline에 time-aware 요소 추가
- 방식:
  - Additive
    - bucket
    - continuous
  - Attention Bias
- 시간 정보:
  - `delta_prev_seconds`
  - `delta_start_seconds`

3. Stage 3
- multitask learning
- 기존 SASRec backbone 위에
  - next activity classification head
  - next time regression head
  를 함께 두는 구조
- next time target:
  - `delta_next_seconds`

## 2. 데이터와 전처리 관련 중요 맥락

### 2-1. 데이터셋
- BPI 2012 complete-only 데이터 사용

### 2-2. 전처리 파일 안정성 점검 결과
- Stage 1 baseline용 interaction 데이터는 정상 확인
- Stage 2에서 사용한 Drive의 `4/20` processed 세트는 정상 확인
- Stage 3 과정에서 한때 로컬 processed 파일이 손상된 적이 있었음
  - mixed timestamp parsing 문제
  - 일부 blank timestamp / blank delta 이슈
- 이후 전처리 코드를 수정하고 Stage 3용 데이터셋을 별도 버전으로 재생성함

### 2-3. Stage 3 데이터셋 버전 관리
- Stage 2 데이터는 기존 폴더 유지
- Stage 3는 별도 폴더로 분리하여 사용
- Stage 3용 processed dataset:
  - `bpi2012_complete_only_stage3_v2`
- 따라서 Stage 3 최종 실험은 수정된 전처리 코드와 versioned dataset 기준으로 다시 수행된 결과임

### 2-4. 전처리 관련 보고서 작성 시 주의점
- Stage 1, Stage 2 결과는 유효한 것으로 판단
- Stage 3는 최종적으로 `stage3_v2` 기준 결과만 사용해야 함

## 3. Stage 1 / Stage 2 핵심 결론

### 3-1. Stage 1 baseline
- `anchor_ml20`, `refine_ml50_do035`를 주요 baseline 후보로 비교
- sanity check까지 종합했을 때 overall best baseline은 `anchor_ml20`

### 3-2. Stage 2 time-aware
- main metric 기준으로 baseline을 명확히 넘는 time-aware 방법은 없었음
- time-aware 방법들 중에서는 Attention Bias가 가장 유망했음
- best time-aware candidate:
  - `refine_ml50_do035 + delta_start_seconds + 9-bucket attention bias`
- 그러나 overall best는 여전히 `anchor_ml20` single-task baseline

## 4. Stage 3 실험 설계

### 4-1. multitask 구조
- shared sequence representation 위에
  - next activity classification
  - next time regression
  를 함께 학습

### 4-2. next time target
- `delta_next_seconds`
- 의미:
  - 현재 이벤트와 다음 이벤트 사이의 시간 차이(초 단위)

### 4-3. 학습 관련 설정
- activity loss:
  - SASRec 기본 방식과 동일
  - positive item / negative sampled item 기반
  - `BCEWithLogitsLoss`
- time loss:
  - `Huber(SmoothL1) loss`
- total loss:
  - `activity loss + time_loss_weight × time loss`
- time target:
  - `log1p(delta_next_seconds)` 변환 후 학습

### 4-4. time_loss_weight 의미
- `1.0`
  - time loss를 기본 비중으로 반영
- `0.1`
  - time loss 비중을 10분의 1로 줄여 next activity를 더 우선

### 4-5. Stage 3 main metric
- activity main metric:
  - `full ranking, NDCG@10`
- time main metric:
  - `MAE`
- 보조 time metric:
  - `RMSE`
  - `Median AE`

## 5. next time 예측에 대한 해석 원칙

### 5-1. `delta_next_seconds`의 위치
- process mining / predictive process monitoring 문헌에서 더 보편적인 time target은 `remaining time`
- 본 연구는 remaining time이 아니라
  - `next event까지의 local time gap`
  인 `delta_next_seconds`를 예측함
- 따라서 다른 논문과 절대 수치 비교는 조심해야 함

### 5-2. time metric 해석 원칙
- `MAE`를 main time metric으로 사용
- 이유:
  - 해석이 가장 직관적
  - 평균적으로 얼마나 틀리는지 직접 설명 가능
- `RMSE`:
  - 큰 오차(outlier)에 더 민감
- `Median AE`:
  - 전형적인 케이스에서의 오차 해석용

## 6. Stage 3에서 사용한 simple predictor

`naive baseline`이라는 표현 대신 `simple predictor`라는 용어를 사용함.

### 6-1. 목적
- next time 성능의 절대 수준을 해석하기 위한 비교 기준

### 6-2. 계산 방식
- train split의 시간 분포에서 단순 통계를 구하고
- 그 값을 test split에 그대로 적용하여
  - MAE
  - RMSE
  - Median AE
  계산

### 6-3. 사용한 simple predictor
- `global_mean`
- `global_median`
- `activity_mean`
- `prefix_length_mean`

### 6-4. 실제 해석에 중요하게 본 predictor
- `global_median`
- `activity_mean`

`prefix_length_mean`은 계산은 했지만 해석에서 핵심적으로 사용하지는 않음

### 6-5. simple predictor 비교를 통해 얻은 결론
- 일부 multitask 설정은 simple predictor보다 더 나은 next time MAE를 보였음
- 다만 그 차이가 크지는 않아, next time 예측이 완전히 무의미한 수준은 아니지만 매우 강하다고 보기도 어려움

## 7. Stage 3 실험 구성 요약

엑셀의 Stage 3 sheet 기준으로는 아래 실험 축이 존재함.

### 7-1. Baseline multitask
- `anchor_single_task`
- `anchor_multi_task_w1.0`
- `anchor_multi_task_w0.1`
- `refine_single_task`
- `refine_multi_task_w1.0`
- `refine_multi_task_w0.1`

### 7-2. Attention Bias multitask
- `anchor_attnbias_single_task`
- `anchor_attnbias_multi_task_w1.0`
- `anchor_attnbias_multi_task_w0.1`
- `refine_attnbias_single_task`
- `refine_attnbias_multi_task_w1.0`
- `refine_attnbias_multi_task_w0.1`

## 8. Stage 3 핵심 결론

### 8-1. baseline multitask
- anchor, refine 모두에서 single-task 대비 multitask의 next activity 성능이 낮아짐
- 즉 baseline multitask 자체는 next activity 성능을 개선하지 못함
- 다만 baseline multitask 중에서는 anchor가 refine보다 더 나은 후보였음

### 8-2. attention bias multitask
- attention bias는 next activity 성능 개선에는 제한적이었음
- backbone별 양상은 다르게 나타남
  - anchor:
    - activity 성능은 낮아지고
    - time 성능은 좋아지는 경향
  - refine:
    - activity 성능은 더 높아질 수 있으나
    - time 성능은 다소 낮아지는 경향
- 즉 attention bias 효과는 backbone에 따라 다르게 나타났지만, overall로는 next activity 개선보다는 next time prediction 쪽에 더 가까운 효과를 보인 것으로 해석

### 8-3. time_loss_weight 조정
- `time_loss_weight = 0.1`으로 낮추면 anchor와 refine 모두에서 next activity 성능이 회복됨
- 반면 next time 성능은 다소 저하됨
- 즉 Stage 3에서는 multitask 구조 자체보다도 loss balance의 영향이 큼

### 8-4. Stage 3 overall
- Stage 3에서는 multitask가 next time 예측은 가능하게 했지만, overall next activity 성능을 single-task보다 더 좋게 만들지는 못했음
- activity 기준 best multitask:
  - `anchor_multi_task_w0.1`
- time 기준 best:
  - `anchor_attnbias_multi_task_w1.0`
- overall best는 여전히 single-task baseline

## 9. 보고서 작성 시 표현 주의

### 9-1. baseline 용어
- Stage 1의 `baseline model`과
- Stage 3의 `simple predictor`
는 구분해서 써야 함
- `naive baseline`이라는 표현은 혼동 가능성이 있어 피하는 것이 좋음

### 9-2. attention bias 해석
- “attention bias가 next activity 성능을 개선했다”라고 일반화하면 안 됨
- backbone에 따라 결과가 달랐음
- 보다 안전한 표현:
  - attention bias는 overall next activity 개선에는 제한적이었고,
  - backbone에 따라 activity/time trade-off 양상이 다르게 나타났음

### 9-3. multitask 의미 해석
- multitask가 next activity main metric을 넘지 못했다고 해서 “무의미”하다고 단정하지는 않음
- 보다 정확한 해석:
  - next activity 단일 성능 향상 방법으로는 제한적이었음
  - 그러나 activity-time trade-off와 loss balance effect를 확인했다는 점에서 연구적 의미가 있음

## 10. 최종 보고서에서 강조 가능한 문장

- Stage 2에서는 attention bias가 가장 유망한 time-aware 방식이었으나, overall best는 여전히 single-task baseline이었다.
- Stage 3에서는 multitask learning을 통해 next time prediction을 함께 학습할 수 있었으나, overall next activity 성능을 single-task보다 더 높이지는 못했다.
- 특히 Stage 3에서는 activity prediction과 time prediction 사이의 trade-off, 그리고 `time_loss_weight` 조정을 통한 loss balance의 영향이 핵심적으로 관찰되었다.
- simple predictor와 비교했을 때 next time prediction은 일정 수준의 의미 있는 성능을 보였지만, 압도적으로 강한 수준이라고 보기는 어려웠다.

## 11. 엑셀 파일과의 관계

최종 엑셀 파일에는 이미 아래 내용이 포함되어 있다고 가정한다.
- Stage 1 / Stage 2 / Stage 3 실험 결과표
- mean/std 요약
- raw run 결과
- Stage 3 개요

따라서 이 문서는 엑셀에 없는 추가 맥락, 즉
- 왜 이런 실험을 했는지
- 각 metric을 어떻게 해석해야 하는지
- Stage 3 결론을 어떤 톤으로 써야 하는지
를 보완하는 용도로 사용하면 된다.
