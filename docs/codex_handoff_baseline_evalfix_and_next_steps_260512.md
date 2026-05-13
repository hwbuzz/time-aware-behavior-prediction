# Codex Handoff: SASRec Baseline Re-run and Next Sanity Check

## Purpose
This document is for another Codex instance that will continue the work on a different computer after the current Colab notebook finishes running.

The immediate next task after the current notebook finishes is:

1. compare the re-run baseline results,
2. choose the single best run,
3. create a sanity-check notebook for that selected run.

---

## Current project context

- Project: `time-aware-behavior-prediction`
- Dataset: `BPI 2012 complete-only`
- Baseline model: SASRec
- Main evaluation:
  - `full ranking`
- Supplementary evaluation:
  - `sampled ranking` with `num_negative_samples=100`
- Metrics:
  - `NDCG@5`, `HR@5`
  - `NDCG@10`, `HR@10`
  - `MRR`

The user is working in Korean and prefers practical, concise guidance.

---

## Important code changes already made

These changes were made locally and pushed by the user to GitHub before the current notebook run.

### 1. Full ranking evaluation bug was fixed

Previously, full ranking candidate construction and target-rank calculation were inconsistent.

Now in [src/sasrec_utils.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/sasrec_utils.py):

- full ranking candidates are built as:
  - `[target_item] + negatives`
- so rank computation correctly refers to the target item at index `0`

### 2. Sampled test protocol was aligned more closely with original SASRec / pmixer style

In sampled evaluation:

- the test sequence still uses `train + valid` as input context,
- but the negative-sampling exclusion source now uses `train[user]` rather than `train + valid`

This was done in [src/sasrec_utils.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/sasrec_utils.py).

### 3. Dataset summary dead code was cleaned up

`summarize_dataset_splits()` now correctly includes time-embedding metadata when relevant.

### 4. Smoke test after fixes

A local smoke test was run after these fixes:

- baseline run completed
- time-aware run completed
- `metrics_summary.json`, `metrics_history.csv`, `sasrec_best.pth`, `sasrec_last.pth` were all saved correctly

So the fixed code path is known to run successfully.

---

## Current Colab notebook being executed

The notebook currently being run is:

- [sasrec_bpi2012_colab_train_evalfix_260512.ipynb](/C:/Users/hyewoo%20choi/Documents/99.%20EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/sasrec_bpi2012_colab_train_evalfix_260512.ipynb)

This notebook was created from:

- [generate_baseline_retrain_colab_notebook.py](/C:/Users/hyewoo%20choi/Documents/99.%20EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/scripts/generate_baseline_retrain_colab_notebook.py)

### Important notebook behavior

The notebook now includes:

```python
%cd /content/time-aware-behavior-prediction
!git pull
```

This is important because the user explicitly requested that future Colab notebooks should not leave `git pull` commented out.

---

## Output directories for the current run

The user decided to separate the re-run results by selection metric:

- `outputs/sasrec_bpi2012_ndcg10`
- `outputs/sasrec_bpi2012_ndcg5`

These are meant to replace older baseline result folders after the evaluation-fix rerun.

The user said they would back up or rename the old output folders manually.

---

## Run naming scheme for the current rerun

For both output directories, the same run names are used:

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

### Mapping to parameter settings

- `anchor_pd_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=200, lr=0.001, dropout=0.2, seed=42
- `anchor_ml100_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=100, lr=0.001, dropout=0.2, seed=42
- `anchor_ml50_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.2, seed=42
- `anchor_ml20_s42`
  - hidden_units=32, num_blocks=2, num_heads=1, maxlen=20, lr=0.001, dropout=0.2, seed=42
- `refine_ml50_do030_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.3, seed=42
- `refine_ml50_do035_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.35, seed=42
- `refine_ml50_do025_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.25, seed=42
- `refine_ml50_do025_s2024`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.25, seed=2024
- `refine_ml50_do030_s2024`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.3, seed=2024
- `refine_ml75_do030_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=75, lr=0.001, dropout=0.3, seed=42
- `refine_ml100_do030_s42`
  - hidden_units=50, num_blocks=2, num_heads=1, maxlen=100, lr=0.001, dropout=0.3, seed=42

---

## How the notebook is organized

The notebook runs both model-selection criteria:

### A. `NDCG@10` selection

- output dir: `outputs/sasrec_bpi2012_ndcg10`
- selection metric:
  - `full_valid_ndcg@10`

### B. `NDCG@5` selection

- output dir: `outputs/sasrec_bpi2012_ndcg5`
- selection metric:
  - `full_valid_ndcg@5`

At the end, the notebook rebuilds result tables directly from run folders instead of relying on `experiment_index.csv`.

That design should be preserved because `experiment_index.csv` has previously suffered from schema drift.

---

## What the next Codex should do after the notebook finishes

### Step 1. Read the completed results

Use the notebook outputs or rebuild from:

- `outputs/sasrec_bpi2012_ndcg10`
- `outputs/sasrec_bpi2012_ndcg5`

Do not assume the old baseline results are still valid, because the evaluation code changed.

### Step 2. Compare the runs separately by selection criterion

The user explicitly wants this rule:

- if best epoch selection was based on `NDCG@10`, interpret primarily with `NDCG@10`
- if best epoch selection was based on `NDCG@5`, interpret primarily with `NDCG@5`

So:

- for `sasrec_bpi2012_ndcg10`, rank/interpret using `best_valid_full_ndcg@10`, `best_test_full_ndcg@10`
- for `sasrec_bpi2012_ndcg5`, rank/interpret using `best_valid_full_ndcg@5`, `best_test_full_ndcg@5`

Supplementary metrics (`sampled`, `MRR`, other top-k) can still be discussed, but the main comparison should follow the selection metric.

### Step 3. Choose one final baseline run

The user wants to pick **one single best run** after comparison.

This means:

- one configuration
- one selection-metric track

The exact winner should be determined after the current notebook completes.

### Step 4. Create a new sanity-check notebook for that chosen run

After the final winner is chosen, create a new Colab notebook that:

- reuses the chosen configuration,
- adds additional seeds,
- compares mean/std,
- checks valid/test consistency.

The user plans to do this in a separate later step.

---

## Sanity-check expectations

Previous sanity-check logic was already documented in:

- [sasrec_sanity_check_plan.md](/C:/Users/hyewoo%20choi/Documents/99.%20EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/docs/sasrec_sanity_check_plan.md)

Core idea:

1. pick final candidate(s),
2. add extra seeds,
3. compare mean/std,
4. check whether validation and test show a consistent trend.

For the next Codex, this should be adapted to the **new re-run baseline winner**, not the older pre-fix results.

---

## Important historical context

Before the evaluation fix:

- full ranking results were unreliable because of the target-rank bug
- sampled test protocol differed from the original SASRec-style setup

So any comparisons done before the fixes should be treated as historical only, not final.

The newly re-run baseline notebook is intended to become the new trustworthy baseline reference.

---

## Practical guidance for the next Codex

When continuing from here:

- first inspect the completed output folders,
- summarize the `ndcg10` and `ndcg5` tracks separately,
- identify the best single run,
- then make a new sanity-check notebook specifically for that run.

It is better to:

- rebuild results from `metrics_summary.json` / `config.json`
- avoid depending directly on legacy `experiment_index.csv`

Also remember:

- for future Colab notebooks, the user wants `git pull` to be active, not commented out.

---

## Suggested phrasing for the next decision point

After the notebook finishes, the next Codex should answer something like:

- which run is best under `NDCG@10` selection?
- which run is best under `NDCG@5` selection?
- if only one final baseline must be kept, which one is the most defensible overall?
- then build the sanity-check notebook for that final choice.

