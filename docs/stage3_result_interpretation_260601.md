# Stage 3 Result Interpretation (260601)

## Scope

This note summarizes the interpretation of the executed results in:

- `notebooks/sasrec_stage3_bpi2012_colab_train_02_260601.ipynb`
- `notebooks/sasrec_stage3_bpi2012_colab_train_03_260601.ipynb`

The primary comparison metric is:

- `best_test_at_best_valid_full_ndcg@10`

Best epoch selection was based on:

- `full_valid_ndcg@10`


## 1. Stage 3 Baseline Multi-Task Result

Notebook:

- `sasrec_stage3_bpi2012_colab_train_02_260601.ipynb`

This notebook compares:

- `anchor_single_task` vs `anchor_multi_task`
- `refine_single_task` vs `refine_multi_task`

### Main result

`full NDCG@10` on test at best valid epoch:

- `anchor_single_task`: **0.8898**
- `anchor_multi_task`: **0.8028**
- `refine_single_task`: **0.8467**
- `refine_multi_task`: **0.6836**

### Interpretation

- Adding the multi-task head degraded next-activity ranking performance for both backbones.
- The drop was about `-0.087` for `anchor`.
- The drop was about `-0.163` for `refine`.
- The performance drop was larger on the `refine` backbone.

This suggests that the current multi-task setup:

- successfully trains joint next-activity and next-time prediction,
- but places a substantial burden on the main next-activity task.

### Time prediction result

Representative test-time errors:

- `anchor_multi_task`
  - `time_mae`: **12674s**
  - `time_rmse`: **72600s**
  - `time_median_ae`: **71.9s**
- `refine_multi_task`
  - `time_mae`: **12810s**
  - `time_rmse`: **76606s**
  - `time_median_ae`: **69.8s**

### Time-metric interpretation

- `MAE` and `RMSE` are very large.
- `Median AE` is much smaller, around about one minute.

This indicates a heavy-tailed error distribution:

- the model is often reasonably close,
- but a smaller number of very large errors increases the average error strongly.

For reporting, `Median AE` should be discussed together with `MAE/RMSE`.


## 2. Stage 3 Attention-Bias Multi-Task Result

Notebook:

- `sasrec_stage3_bpi2012_colab_train_03_260601.ipynb`

This notebook compares the following variants on the `anchor` backbone:

- `anchor_single_task`
- `anchor_attnbias_single_task`
- `anchor_multi_task`
- `anchor_attnbias_multi_task`

Important note:

- This is an `anchor`-backbone attention-bias comparison.
- It is not the `refine`-backbone best attention-bias family from Stage 2.

### Main result

`full NDCG@10` on test at best valid epoch:

- `anchor_single_task`: **0.8898**
- `anchor_attnbias_single_task`: **0.8355**
- `anchor_multi_task`: **0.8028**
- `anchor_attnbias_multi_task`: **0.7428**

### Interpretation

- On the `anchor` backbone, attention bias itself did not outperform the baseline.
- Extending the attention-bias model to the multi-task setting further reduced next-activity performance.

Therefore, under the current setup:

- attention bias does not recover the activity-performance drop caused by multi-task learning,
- and the combination of `attention bias + multi-task` is weaker than the corresponding single-task variants.

### Time prediction result

Comparison between the two multi-task variants:

- `anchor_multi_task`
  - `time_mae`: **12674s**
  - `time_rmse`: **72600s**
  - `time_median_ae`: **71.9s**
- `anchor_attnbias_multi_task`
  - `time_mae`: **11937s**
  - `time_rmse`: **71007s**
  - `time_median_ae`: **100.6s**

### Time-metric interpretation

- `anchor_attnbias_multi_task` is slightly better in `MAE/RMSE`.
- However, its `Median AE` is worse than `anchor_multi_task`.

This suggests:

- attention bias may reduce some larger outlier errors,
- but it does not improve the typical-case timing error consistently.


## 3. Overall Stage 3 Conclusion

The overall conclusion from the current Stage 3 experiments is:

- The multi-task extension is technically successful: the model can jointly predict next activity and next time.
- However, under the current setting, it does not improve the primary next-activity objective.
- On the contrary, `full NDCG@10` shows a consistent degradation relative to the single-task baselines.
- Time prediction is feasible, but its benefit is not strong enough to offset the loss in next-activity performance.
- On the `anchor` backbone, adding attention bias in the multi-task setting does not recover the performance drop.

A concise interpretation is:

> Stage 3 multi-task extension is feasible, but under the current configuration it is not effective for improving next-activity prediction.


## 4. Recommended Interpretation for Writing

The following points are reasonable to state in the thesis or report:

- The Stage 3 architecture successfully enabled joint next-activity and next-time prediction.
- Nevertheless, the multi-task objective did not outperform the single-task baselines on the main next-activity metric.
- The auxiliary time-prediction head appears to interfere with the shared representation needed for next-activity ranking.
- Attention bias did not provide additional gains in the current multi-task setting on the `anchor` backbone.


## 5. Next Step

The current results provide a strong motivation for the next experiment:

- `attention bias multi-task + smaller time_loss_weight`

Recommended candidates:

- `time_loss_weight = 0.5`
- `time_loss_weight = 0.1`

Reason:

- The current results suggest that `time_loss_weight = 1.0` may be too strong for the main next-activity task.
- The issue may be not that multi-task learning is inherently ineffective, but that the loss balance is too aggressive.
