# Stage 3 Experiment Plan

## Goal

Stage 3 extends the current SASRec-based next-activity prediction setup into a multi-task setting that predicts:

- next activity
- next time

The main research question is whether adding a next-time prediction objective can improve or at least preserve next-activity prediction performance, and whether the most promising time-aware backbone from Stage 2 remains effective in the multi-task setting.

## Existing Reference Runs

The following Stage 1 and Stage 2 runs will be used as comparison baselines.

### Stage 1 baseline reference

- `anchor_ml20`
  - final strongest baseline from Stage 1

### Additional baseline reference

- `refine_ml50_do035`
  - baseline backbone used for the strongest attention-bias family in Stage 2

### Stage 2 time-aware reference

- `attnbias_dstart_ml50_do035_b9`
  - strongest time-aware single-task result from Stage 2

## Stage 3 New Experiments

### 1. Baseline multi-task

- input: `activity sequence`
- output: `next activity + next time`
- purpose: verify the effect of multi-task learning itself

### 2. Attention bias multi-task

- input: `activity sequence + delta_start attention bias`
- output: `next activity + next time`
- purpose: verify whether the most effective time-aware backbone from Stage 2 is still effective in Stage 3

### 3. Attention bias multi-task with different `time_loss_weight` (optional)

- example: compare default `1.0` with one additional value such as `0.1` or `0.5`
- purpose: check whether mixed results come from poor loss balance rather than from the multi-task setting itself

## Comparison Structure

### A. Multi-task effect

- `anchor_ml20` vs `baseline multi-task`
- `refine_ml50_do035` vs `baseline multi-task`

Question:

- Does adding next-time prediction help or harm next-activity prediction?

### B. Attention bias effect

- `refine_ml50_do035` vs `attnbias_dstart_ml50_do035_b9`
- `baseline multi-task` vs `attention bias multi-task`

Question:

- Is attention bias effective not only in the single-task setting but also in the multi-task setting?

### C. Multi-task extension on the attention-bias backbone

- `attnbias_dstart_ml50_do035_b9` vs `attention bias multi-task`

Question:

- What changes when the Stage 2 best attention-bias model is extended from single-task to multi-task?

### D. Loss-balance effect (optional)

- `attention bias multi-task (time_loss_weight=1.0)` vs `attention bias multi-task (time_loss_weight=0.1 or 0.5)`

Question:

- If results are weak or unstable, is the issue caused by loss balance rather than the model idea itself?

## Best Epoch Criterion

All Stage 3 experiments will use the same best-epoch criterion:

- `full_valid_ndcg@10`

Reason:

- next-activity prediction remains the main task
- this keeps Stage 3 directly comparable with Stage 1 and Stage 2
- time prediction metrics will still be recorded and analyzed at the selected best epoch

## Metrics To Save And Print

All metrics currently defined in the code should be saved and printed.

### Ranking metrics

- `full ndcg@5`
- `full ndcg@10`
- `full hr@5`
- `full hr@10`
- `full mrr`
- `sampled ndcg@5`
- `sampled ndcg@10`
- `sampled hr@5`
- `sampled hr@10`
- `sampled mrr`

### Shared classification metrics

- `accuracy`
- `macro_f1`
- `top5_accuracy`
- `top10_accuracy`

### Shared time-prediction metrics

- `time_mae`
- `time_rmse`
- `time_median_ae`

## Recommended Scope

To avoid running too many experiments, the recommended Stage 3 scope is:

- required:
  - `baseline multi-task`
  - `attention bias multi-task`
- optional:
  - one additional `attention bias multi-task` run with a different `time_loss_weight`

This scope is considered sufficient to support the Stage 3 research story.
