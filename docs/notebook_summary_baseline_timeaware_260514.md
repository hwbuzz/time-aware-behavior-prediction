# Notebook Summary

이 문서는 baseline 재실행, sanity check, time-aware 실험용 Colab notebook들의 역할을 빠르게 구분하기 위한 요약 문서이다.

## Baseline / Sanity Check

### `sasrec_bpi2012_colab_train_evalfix_260512.ipynb`
- evaluation fix 이후 **baseline SASRec run들을 다시 학습**하는 notebook.
- 목적:
  - full ranking bug fix
  - sampled test protocol fix
  가 반영된 코드 기준으로 baseline 성능을 다시 확보하는 것.
- output dir:
  - `outputs/sasrec_bpi2012_ndcg10`
  - `outputs/sasrec_bpi2012_ndcg5`
- 주요 run:
  - `anchor_pd_s42`
  - `anchor_ml100_s42`
  - `anchor_ml50_s42`
  - `anchor_ml20_s42`
  - `refine_ml50_do030_s42`
  - `refine_ml50_do035_s42`
  - `refine_ml50_do025_s42`
  - `refine_ml50_do025_s2024`
  - `refine_ml50_do030_s2024`
  - `refine_ml75_do030_s42`
  - `refine_ml100_do030_s42`

### `sasrec_bpi2012_colab_train_sanity_check_evalfix_260513.ipynb`
- evaluation fix 이후 다시 얻은 baseline 결과를 바탕으로 **최종 후보 2개에 대한 sanity check**를 수행하는 notebook.
- 목적:
  - seed variation 확인
  - reproducibility 확인
  - valid/test 경향 확인
- 비교 후보:
  - `anchor_ml20`
  - `refine_ml50_do035`
- 이 notebook 결과를 바탕으로 **최종 baseline은 `anchor_ml20`로 선택**되었다.

## Time-Aware

### `sasrec_timeaware_bpi2012_colab_train_02_260513.ipynb`
- **최종 baseline `anchor_ml20`** 위에 additive time embedding을 붙여보는 첫 time-aware notebook.
- 목적:
  - `delta_prev_seconds`
  - bucketized time embedding
  - `8-bucket`, `9-bucket`
  를 baseline과 비교하는 것.
- 구조:
  - baseline 재사용
  - 새로 학습:
    - `time-aware b8`
    - `time-aware b9`
- 결론:
  - `anchor_ml20` baseline이 여전히 가장 강했고,
  - additive bucket time-aware는 baseline을 넘지 못했다.

### `sasrec_timeaware_bpi2012_colab_train_03_260513.ipynb`
- **`refine_ml50_do035` baseline** 위에 additive bucket time embedding을 붙여보는 notebook.
- 목적:
  - `delta_prev_seconds`
  - bucketized time embedding
  - `8-bucket`, `9-bucket`
  을 `refine_ml50_do035` baseline과 비교하는 것.
- 구조:
  - baseline 재사용
  - 새로 학습:
    - `time-aware b8`
    - `time-aware b9`
- 주요 해석:
  - validation에서는 일부 개선 신호가 있었지만,
  - test에서는 baseline을 넘지 못했다.

### `sasrec_timeaware_bpi2012_colab_train_04_260514.ipynb`
- **`refine_ml50_do035` baseline** 위에 additive **continuous/log-delta time embedding**을 붙여보는 notebook.
- 목적:
  - `delta_prev_seconds`
  - continuous time encoding
  - `[log1p(delta_prev_seconds), is_first_event] -> projection`
  구조를 baseline과 비교하는 것.
- 구조:
  - baseline 재사용
  - 새로 학습:
    - continuous time-aware
- 주요 해석:
  - validation fit은 bucket보다 약간 좋아 보였지만,
  - test generalization과 안정성은 baseline보다 낫지 않았다.

### `sasrec_timeaware_bpi2012_colab_train_05_260514.ipynb`
- **`refine_ml50_do035` baseline** 위에서 `delta_start_seconds`를 사용하는 same-framework 재실험 notebook.
- 목적:
  - `delta_start_seconds + 9-bucket`
  - `delta_start_seconds + continuous`
  를 같은 baseline 위에서 비교하는 것.
- 구조:
  - baseline 재사용
  - 새로 학습:
    - `delta_start + 9-bucket`
    - `delta_start + continuous`
- 의미:
  - 기존 `delta_prev_seconds` 기반 실험과 달리,
  - **case 시작 이후 누적 시간**이 더 도움이 되는지 보는 실험.

### `sasrec_timeaware_bpi2012_colab_train_06_260514.ipynb`
- **attention bias 방식**을 처음 실험하기 위한 notebook.
- 목적:
  - baseline `refine_ml50_do035` 재사용
  - `delta_start_seconds`
  - causal pairwise gap
  - `9-bucket attention bias`
  를 사용하는 time-aware 실험 수행
- 구조:
  - baseline 재사용
  - 새로 학습:
    - `attnbias_dstart_ml50_do035_b9_s42`
    - `attnbias_dstart_ml50_do035_b9_s2024`
    - `attnbias_dstart_ml50_do035_b9_s7`
- output dir:
  - `outputs/sasrec_timeaware_attention_bias_ndcg10`
  - `outputs/sasrec_timeaware_attention_bias_ndcg5`
- 의미:
  - 기존 additive time embedding이 아니라,
  - **attention score에 time bias를 직접 반영하는 첫 구조 변경 실험**이다.

### `sasrec_timeaware_bpi2012_colab_train_07_260514.ipynb`
- **최종 baseline `anchor_ml20`** 위에 attention bias 방식을 적용해보는 notebook.
- 목적:
  - baseline `anchor_ml20` 재사용
  - `delta_start_seconds`
  - causal pairwise gap
  - `9-bucket attention bias`
  를 strongest baseline 위에서 검증하는 것.
- 구조:
  - baseline 재사용
  - 새로 학습:
    - `attnbias_dstart_ml20_b9_s42`
    - `attnbias_dstart_ml20_b9_s2024`
    - `attnbias_dstart_ml20_b9_s7`
- output dir:
  - `outputs/sasrec_timeaware_attention_bias_ndcg10`
  - `outputs/sasrec_timeaware_attention_bias_ndcg5`
- 의미:
  - `refine_ml50_do035` 위에서 유망하게 보였던 attention bias를,
  - **최종 baseline `anchor_ml20`에도 적용했을 때 improvement가 있는지** 확인하는 실험이다.

## 한 줄 정리

- `evalfix_260512`: baseline 전체 재학습
- `sanity_check_evalfix_260513`: baseline 최종 후보 sanity check
- `02`: `anchor_ml20` + bucket time-aware
- `03`: `refine_ml50_do035` + bucket time-aware
- `04`: `refine_ml50_do035` + continuous time-aware
- `05`: `refine_ml50_do035` + `delta_start` bucket/continuous 비교
- `06`: `refine_ml50_do035` + attention bias time-aware
- `07`: `anchor_ml20` + attention bias time-aware
