# Notebook Summary

이 문서는 baseline, sanity check, time-aware, Stage 3 multi-task Colab notebook들의 역할을 빠르게 구분하기 위한 요약 문서이다.

## Baseline / Sanity Check

### `sasrec_bpi2012_colab_train_evalfix_260512.ipynb`
- evaluation fix 이후 baseline SASRec run들을 다시 학습하는 notebook.
- 목적:
  - full ranking bug fix
  - sampled test protocol fix
  가 반영된 코드 기준으로 baseline 성능을 다시 확인하는 것
- output dir:
  - `outputs/sasrec_bpi2012_ndcg10`
  - `outputs/sasrec_bpi2012_ndcg5`

### `sasrec_bpi2012_colab_train_sanity_check_evalfix_260513.ipynb`
- evaluation fix 이후 다시 얻은 baseline 결과를 바탕으로 최종 후보 2개에 대해 sanity check를 수행하는 notebook.
- 목적:
  - seed variation 확인
  - reproducibility 확인
  - valid/test 경향 확인
- 비교 후보:
  - `anchor_ml20`
  - `refine_ml50_do035`
- 이 notebook 결과를 바탕으로 최종 baseline은 `anchor_ml20`으로 선정되었다.

## Time-Aware

### `sasrec_timeaware_bpi2012_colab_train_02_260513.ipynb`
- 최종 baseline `anchor_ml20` 위에 additive bucket time embedding을 붙여보는 첫 time-aware notebook.
- 목적:
  - `delta_prev_seconds`
  - bucketized time embedding
  - `8-bucket`, `9-bucket`
  을 baseline과 비교하는 것
- 결론:
  - `anchor_ml20` baseline이 워낙 강했고
  - additive bucket time-aware는 baseline을 넘지 못했다.

### `sasrec_timeaware_bpi2012_colab_train_03_260513.ipynb`
- `refine_ml50_do035` baseline 위에 additive bucket time embedding을 붙여보는 notebook.
- 목적:
  - `delta_prev_seconds`
  - bucketized time embedding
  - `8-bucket`, `9-bucket`
  을 `refine_ml50_do035` baseline과 비교하는 것
- 결론:
  - validation에서는 일부 개선 신호가 있었지만
  - test에서는 baseline을 넘지 못했다.

### `sasrec_timeaware_bpi2012_colab_train_04_260514.ipynb`
- `refine_ml50_do035` baseline 위에 additive continuous time embedding을 붙여보는 notebook.
- 목적:
  - `delta_prev_seconds`
  - continuous time encoding
  - `[log1p(delta_prev_seconds), is_first_event] -> projection`
  구조를 baseline과 비교하는 것
- 결론:
  - validation fit은 일부 좋아졌지만
  - test generalization과 안정성은 baseline보다 낫지 않았다.

### `sasrec_timeaware_bpi2012_colab_train_05_260514.ipynb`
- `refine_ml50_do035` baseline 위에서 `delta_start_seconds`를 사용하는 same-framework 실험 notebook.
- 목적:
  - `delta_start_seconds + 9-bucket`
  - `delta_start_seconds + continuous`
  를 같은 baseline 위에서 비교하는 것
- 의미:
  - 기존 `delta_prev_seconds` 기반 실험과 달리
  - case 시작 이후 누적 시간을 쓰면 달라지는지 보는 실험

### `sasrec_timeaware_bpi2012_colab_train_06_260514.ipynb`
- attention bias 방식을 처음 실험하기 위한 notebook.
- 목적:
  - baseline `refine_ml50_do035`
  - `delta_start_seconds`
  - causal pairwise gap
  - `9-bucket attention bias`
  를 사용하는 time-aware 실험 수행
- 결론:
  - Stage 2 time-aware 중 가장 유망한 후보
  - best Stage 2 time-aware candidate는 `refine_ml50_do035 + delta_start + b9 attention bias`

### `sasrec_timeaware_bpi2012_colab_train_07_260514.ipynb`
- 최종 baseline `anchor_ml20` 위에 attention bias 방식을 적용해보는 notebook.
- 목적:
  - strongest baseline인 `anchor_ml20`에서도
  - `delta_start_seconds + 9-bucket attention bias`
  가 유효한지 확인하는 것
- 결론:
  - additive time-aware보다는 나았지만
  - `anchor_ml20` baseline 자체를 넘지는 못했다.

### `sasrec_timeaware_bpi2012_colab_train_08_260515.ipynb`
- `refine_ml50_do035` baseline 위에서 attention bias bucket 경계를 한 번 더 세분화해보는 refinement notebook.
- 목적:
  - 기존 `b9`
  - 새 `b10`
  을 baseline과 함께 비교하는 것
- 결론:
  - `b10`은 `b9`를 넘지 못했고
  - Stage 2 attention bias best setting은 여전히 `b9`였다.

## Stage 3 Multi-Task

### `sasrec_stage3_bpi2012_colab_train_01_260531.ipynb`
- Stage 3의 첫 baseline multi-task pilot notebook.
- 목적:
  - `anchor_ml20`
  - `refine_ml50_do035`
  backbone을 기준으로
  - `next activity + next time`
  를 함께 예측하는 baseline multi-task 설정이 돌아가는지 먼저 확인하는 것
- 주의:
  - 초기 pilot notebook이라 현재 권장 Stage 3 데이터 세트(`bpi2012_complete_only_stage3_v2`) 기준은 아님
  - 실질적 기준은 `02` 이후 notebook들

### `sasrec_stage3_bpi2012_colab_train_02_260601.ipynb`
- Stage 3 baseline multi-task의 3-seed 본 비교 notebook.
- 목적:
  - single-task baseline 3-seed 결과 재사용
  - baseline multi-task 3-seed 결과 비교
  - mean/std 기준으로 Stage 3 baseline multi-task 효과 해석
- 비교 대상:
  - `anchor_single_task`
  - `refine_single_task`
  - `anchor_multi_task`
  - `refine_multi_task`
- 결론:
  - multi-task는 next-time 예측은 가능하게 했지만
  - main task인 next-activity ranking은 전반적으로 떨어뜨렸다.

### `sasrec_stage3_bpi2012_colab_train_03_260601.ipynb`
- Stage 3에서 `anchor_ml20` backbone 기반 attention-bias multi-task를 실험하는 notebook.
- 목적:
  - Stage 2 `anchor` single-task attention bias 결과
  - Stage 3 baseline multi-task 결과
  를 연결해서
  - `anchor + attention bias + multi-task`
  가 추가 이득이 있는지 확인하는 것
- 비교 대상:
  - `anchor_single_task`
  - `anchor_attnbias_single_task`
  - `anchor_multi_task`
  - `anchor_attnbias_multi_task`
- 결론:
  - attention bias를 multi-task에 붙여도 ranking은 더 나빠졌고
  - 일부 next-time MAE/RMSE만 약간 좋아졌다.

### `sasrec_stage3_bpi2012_colab_train_04_260601.ipynb`
- Stage 3에서 `anchor_ml20 + attention-bias multi-task`에 `time_loss_weight=0.1`을 적용해보는 notebook.
- 목적:
  - `anchor_attnbias_multi_task_w1.0`
  - `anchor_attnbias_multi_task_w0.1`
  을 비교해서
  - time loss 비중 조정이 ranking 회복에 도움이 되는지 확인하는 것
- 비교 대상:
  - `anchor_single_task`
  - `anchor_attnbias_single_task`
  - `anchor_multi_task`
  - `anchor_attnbias_multi_task_w1.0`
  - `anchor_attnbias_multi_task_w0.1`
- 결론:
  - `w0.1`이 `w1.0`보다는 ranking을 일부 회복했지만
  - plain `anchor_multi_task`는 넘지 못했다.

### `sasrec_stage3_bpi2012_colab_train_05_260602.ipynb`
- Stage 3에서 plain `anchor_ml20` multi-task에 `time_loss_weight=0.1`을 적용해보는 notebook.
- 목적:
  - `anchor_multi_task_w1.0`
  - `anchor_multi_task_w0.1`
  을 비교해서
  - plain multitask의 성능 저하가 loss balance 문제인지 확인하는 것
- 비교 대상:
  - `anchor_single_task`
  - `anchor_multi_task_w1.0`
  - `anchor_multi_task_w0.1`
- 결론:
  - `w0.1`로 낮추자 ranking 성능이 크게 회복되었고
  - 현재 Stage 3 best multitask는 `anchor_multi_task_w0.1`이 되었다.

### `sasrec_stage3_bpi2012_colab_train_06_260602.ipynb`
- Stage 3에서 plain `refine_ml50_do035` multi-task에 `time_loss_weight=0.1`을 적용해보는 notebook.
- 목적:
  - `refine_multi_task_w1.0`
  - `refine_multi_task_w0.1`
  비교를 통해
  - `anchor`에서 보인 loss-balance 효과가 `refine`에서도 재현되는지 확인하는 것
- 비교 대상:
  - `refine_single_task`
  - `refine_multi_task_w1.0`
  - `refine_multi_task_w0.1`
- 의미:
  - Stage 3에서 loss-weight 조정 효과가 backbone 전반에 일반화되는지 확인하는 보조 실험

## 한 줄 정리

- `evalfix_260512`: baseline 전체 재학습
- `sanity_check_evalfix_260513`: baseline 최종 후보 sanity check
- `02`: `anchor_ml20` + bucket time-aware
- `03`: `refine_ml50_do035` + bucket time-aware
- `04`: `refine_ml50_do035` + continuous time-aware
- `05`: `refine_ml50_do035` + `delta_start` bucket/continuous 비교
- `06`: `refine_ml50_do035` + attention bias time-aware
- `07`: `anchor_ml20` + attention bias time-aware
- `08`: `refine_ml50_do035` + attention bias bucket-boundary refinement
- `stage3_01`: baseline multi-task pilot
- `stage3_02`: baseline multi-task 3-seed 본 비교
- `stage3_03`: `anchor_ml20` + attention-bias multi-task
- `stage3_04`: `anchor_ml20` + attention-bias multi-task `time_loss_weight=0.1`
- `stage3_05`: `anchor_ml20` + plain multi-task `time_loss_weight=0.1`
- `stage3_06`: `refine_ml50_do035` + plain multi-task `time_loss_weight=0.1`
