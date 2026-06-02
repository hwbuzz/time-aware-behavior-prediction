# Stage 3 Multi-Task Results Summary

이 문서는 Stage 3 multi-task 실험 결과를 한 번에 정리하기 위한 요약 문서이다.

기본 비교 원칙:
- main activity metric: `full ranking + NDCG@10`
- main time metric: `MAE`
- 보조 time metric:
  - `RMSE`
  - `median AE`

Stage 3의 기본 질문:
- `next activity + next time`을 함께 예측하는 multi-task learning이 next-activity prediction에 도움이 되는가?
- 도움이 되지 않는다면, 그 원인은 backbone 문제인가, attention-bias 문제인가, 아니면 time loss balance 문제인가?

## 1. 실험 목록

### `sasrec_stage3_bpi2012_colab_train_02_260601.ipynb`
- baseline multi-task 3-seed 비교
- 비교 대상:
  - `anchor_single_task`
  - `refine_single_task`
  - `anchor_multi_task`
  - `refine_multi_task`

### `sasrec_stage3_bpi2012_colab_train_03_260601.ipynb`
- `anchor` backbone에서 attention-bias multi-task 비교
- 비교 대상:
  - `anchor_single_task`
  - `anchor_attnbias_single_task`
  - `anchor_multi_task`
  - `anchor_attnbias_multi_task`

### `sasrec_stage3_bpi2012_colab_train_04_260601.ipynb`
- `anchor_attnbias_multi_task`에서 `time_loss_weight=1.0 -> 0.1` 비교
- 비교 대상:
  - `anchor_attnbias_multi_task_w1.0`
  - `anchor_attnbias_multi_task_w0.1`

### `sasrec_stage3_bpi2012_colab_train_05_260602.ipynb`
- plain `anchor_multi_task`에서 `time_loss_weight=1.0 -> 0.1` 비교
- 비교 대상:
  - `anchor_multi_task_w1.0`
  - `anchor_multi_task_w0.1`

### `sasrec_stage3_bpi2012_colab_train_06_260602.ipynb`
- plain `refine_multi_task`에서 `time_loss_weight=1.0 -> 0.1` 비교
- 비교 대상:
  - `refine_multi_task_w1.0`
  - `refine_multi_task_w0.1`

### `sasrec_stage3_bpi2012_time_naive_baselines_07_260602.ipynb`
- `delta_next_seconds`에 대한 naive baseline 계산
- baseline:
  - `global_mean`
  - `global_median`
  - `activity_mean`
  - `prefix_len_mean`

## 2. 핵심 결과

## 2-1. multi-task 자체는 next-activity 성능을 낮췄다

`02` 기준 main metric test mean:
- `anchor_single_task`: `0.8898 ± 0.0426`
- `anchor_multi_task_w1.0`: `0.8028 ± 0.0043`
- `refine_single_task`: `0.8467 ± 0.0621`
- `refine_multi_task_w1.0`: `0.7112 ± 0.0380`

해석:
- `anchor`, `refine` 모두에서 single-task가 multi-task보다 더 좋았다.
- 따라서 현재 설정에서는 multi-task learning이 next-activity ranking을 직접 향상시키지는 못했다.

## 2-2. attention bias를 multi-task에 붙여도 ranking은 더 좋아지지 않았다

`03` 기준 main metric test mean:
- `anchor_multi_task_w1.0`: `0.8028 ± 0.0043`
- `anchor_attnbias_multi_task_w1.0`: `0.7428 ± 0.0683`

해석:
- Stage 2에서 유망했던 attention bias도 Stage 3 multi-task에서는 ranking 손실을 줄이지 못했다.
- 오히려 plain `anchor_multi_task`보다 ranking은 더 낮았다.

## 2-3. 핵심 문제는 backbone보다도 loss balance였다

`04`, `05`, `06`에서 공통적으로 관찰된 점:
- `time_loss_weight`를 `1.0 -> 0.1`로 낮추면
- next-activity ranking이 분명히 회복되었다.

대표 수치:

`anchor` plain multi-task:
- `w1.0`: `0.8028 ± 0.0043`
- `w0.1`: `0.8572 ± 0.0923`

`refine` plain multi-task:
- `w1.0`: `0.7112 ± 0.0380`
- `w0.1`: `0.8126 ± 0.0393`

`anchor` attention-bias multi-task:
- `w1.0`: `0.7428 ± 0.0683`
- `w0.1`: `0.7792 ± 0.1391`

해석:
- Stage 3의 성능 저하는 “multi-task 자체가 무조건 나쁘다”기보다
- `next time` loss가 너무 강하게 들어가면서 main activity task를 침식한 영향이 컸다고 볼 수 있다.

## 2-4. 그래도 single-task baseline은 넘지 못했다

가장 좋은 Stage 3 multi-task는 `anchor_multi_task_w0.1`이었지만,
- `anchor_single_task`: `0.8898 ± 0.0426`
- `anchor_multi_task_w0.1`: `0.8572 ± 0.0923`

즉:
- ranking을 꽤 회복했어도
- single-task baseline을 넘지는 못했다.

따라서 Stage 3의 최종 메시지는:
- multi-task가 의미가 전혀 없는 것은 아니지만
- next-activity 단일 성능 향상용 방법으로는 제한적이었다.

## 3. next-time 성능 해석

main time metric은 `MAE`로 두고 해석한다.

이유:
- 해석이 가장 직관적이다.
- “평균적으로 얼마나 틀리느냐”를 보여준다.
- `RMSE`보다 outlier에 덜 끌리고,
- `median AE`보다 전체 성능 대표성이 높다.

## 3-1. Stage 3 model들의 test MAE

- `anchor_attnbias_multi_task_w1.0`: `11937.2 ± 375.7`
- `anchor_multi_task_w1.0`: `12674.0 ± 1361.5`
- `refine_multi_task_w1.0`: `12810.0 ± 1112.7`
- `anchor_attnbias_multi_task_w0.1`: `13225.0 ± 258.6`
- `refine_multi_task_w0.1`: `13306.8 ± 1010.5`
- `anchor_multi_task_w0.1`: `14942.8 ± 1499.6`

시간으로 바꾸면:
- best MAE는 약 `3.3시간`
- worst MAE는 약 `4.15시간`

해석:
- next-time 예측은 완전히 무의미한 수준은 아니다.
- 다만 ranking을 가장 잘 살린 `anchor_multi_task_w0.1`은 time MAE가 가장 나빠졌다.
- 즉 activity와 time 사이의 trade-off가 분명히 존재한다.

## 3-2. naive baseline과 비교

`07` 기준 test naive baseline:
- `global_mean`: `MAE 80778.1`
- `global_median`: `MAE 13491.8`
- `activity_mean`: `MAE 21945.3`
- `prefix_len_mean`: `MAE 76546.1`

해석:
- 강한 naive baseline은 `global_median`이었다.
- `activity_mean`은 MAE보다는 `RMSE`, `median AE` 기준에서 강한 baseline으로 볼 수 있다.

모델과 비교하면:
- `anchor_attnbias_multi_task_w1.0`: `11937.2` → `global_median`보다 좋음
- `anchor_multi_task_w1.0`: `12674.0` → `global_median`보다 좋음
- `refine_multi_task_w1.0`: `12810.0` → `global_median`보다 좋음
- `anchor_attnbias_multi_task_w0.1`: `13225.0` → `global_median`보다 약간 좋음
- `refine_multi_task_w0.1`: `13306.8` → `global_median`보다 약간 좋음
- `anchor_multi_task_w0.1`: `14942.8` → `global_median`보다 나쁨

즉:
- next-time head는 trivial predictor를 대부분 넘는다.
- 하지만 아주 강한 수준이라고 보기는 어렵고,
- 특히 activity를 강하게 살린 설정에서는 time MAE가 다시 약해진다.

## 4. Stage 3 best model 정리

### next-activity main metric 기준 best multi-task
- `anchor_multi_task_w0.1`
- test `NDCG@10 = 0.8572 ± 0.0923`

### next-time MAE 기준 best multi-task
- `anchor_attnbias_multi_task_w1.0`
- test `MAE = 11937.2 ± 375.7`

### overall best single-task
- `anchor_single_task`
- test `NDCG@10 = 0.8898 ± 0.0426`

즉:
- activity 중심으로 보면 `anchor_multi_task_w0.1`
- time 중심으로 보면 `anchor_attnbias_multi_task_w1.0`
- overall next-activity winner는 여전히 `anchor_single_task`

## 5. 최종 결론

1. Stage 3에서 multi-task는 next-time prediction capability를 추가해주지만, 기본 설정에서는 next-activity ranking 성능을 떨어뜨렸다.
2. 이 성능 저하의 핵심 원인은 backbone 자체보다도 `time_loss_weight`가 너무 크게 설정된 loss balance 문제였다고 해석할 수 있다.
3. `time_loss_weight`를 낮추면 ranking 성능은 크게 회복되지만, 그 대가로 next-time MAE는 악화된다.
4. 따라서 Stage 3에서는 activity와 time prediction 사이의 명확한 trade-off가 관찰되었다.
5. 현재까지의 결과를 종합하면, multi-task는 “next-activity 성능을 높이는 방법”으로는 제한적이었지만, joint prediction 구조와 loss-balance trade-off를 분석했다는 점에서 연구적 의미가 있다.
