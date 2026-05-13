# Codex Handoff: Time-Aware SASRec After Final Baseline Selection

## Purpose
This document is for a future Codex instance that will create and/or run the next Colab notebook **after one final baseline SASRec run has been chosen through sanity check**.

That future task is:

1. take the single final baseline run selected after sanity check,
2. keep that baseline fixed,
3. run time-aware SASRec experiments on top of that fixed baseline configuration,
4. compare baseline vs time-aware variants.

---

## Big picture

The project progression is:

1. build and verify a faithful SASRec baseline,
2. fix evaluation issues,
3. re-run baseline experiments,
4. choose one final baseline run through sanity check,
5. then test whether adding time information improves that final baseline.

This document is about **Step 5**.

---

## Current status before the future time-aware step

### Baseline rerun status

A new baseline rerun notebook was prepared:

- [sasrec_bpi2012_colab_train_evalfix_260512.ipynb](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/sasrec_bpi2012_colab_train_evalfix_260512.ipynb)

This notebook re-runs selected baseline configurations using the **fixed evaluation code**.

Output directories for that rerun are:

- `outputs/sasrec_bpi2012_ndcg10`
- `outputs/sasrec_bpi2012_ndcg5`

The future Codex should first inspect the completed results from those folders, then determine the single final baseline run after sanity check.

---

## Important evaluation fixes already applied

These code fixes are already implemented and pushed by the user before the baseline rerun:

### 1. Full ranking bug fix

In [src/sasrec_utils.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/sasrec_utils.py):

- full ranking candidates are now built as:
  - `[target_item] + negatives`
- so the rank is computed for the actual target item

### 2. Sampled test protocol alignment

Also in [src/sasrec_utils.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/sasrec_utils.py):

- sampled test evaluation still uses `train + valid` as the input sequence
- but the negative-sampling exclusion source uses `train[user]`

This is closer to original SASRec / pmixer behavior.

### 3. Smoke test after fixes

After the fixes, a local smoke test was run and both:

- baseline SASRec
- time-aware SASRec

completed successfully with valid/test metrics and checkpoints saved.

So the time-aware code path is known to run.

---

## Time-aware implementation that already exists

The time-aware extension has already been implemented in the codebase.

Relevant files:

- [src/train_sasrec.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/train_sasrec.py)
- [src/sasrec_utils.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/sasrec_utils.py)
- [src/sasrec_model.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/sasrec_model.py)

### Time-aware design

The intended structure is:

```python
x = item_embedding + positional_embedding + time_embedding
```

where:

- `time_embedding` is derived from `delta_prev_seconds`
- `delta_prev_seconds` is bucketized
- the bucket index is passed through an embedding layer

This was intentionally designed so that:

- baseline SASRec can still be run unchanged
- time-aware SASRec can be turned on optionally

### Key CLI options already available

- `--use_time_embedding`
- `--time_features_path`
- `--time_delta_column`
- `--time_bucket_boundaries`
- `--time_bucket_first_event_separate`
- `--time_bucket_zero_gap_separate`

If `--use_time_embedding` is **not** passed, the model behaves like baseline SASRec.

---

## Bucketization design already discussed

The project has already examined the empirical distribution of `delta_prev_seconds`.

### Recommended bucket strategies already considered

#### 8-bucket style

With boundaries:

- `60,600,3600,86400`

and with:

- separate padding bucket
- separate first-event bucket
- separate zero-gap bucket

this corresponds to:

- padding
- first event
- zero-gap
- `(0, 1 min)`
- `[1 min, 10 min)`
- `[10 min, 1 hr)`
- `[1 hr, 1 day)`
- `[>= 1 day)`

#### 9-bucket style

With boundaries:

- `60,600,3600,86400,604800`

and the same special buckets, this corresponds to:

- padding
- first event
- zero-gap
- `(0, 1 min)`
- `[1 min, 10 min)`
- `[10 min, 1 hr)`
- `[1 hr, 1 day)`
- `[1 day, 7 days)`
- `[>= 7 days)`

---

## Previous time-aware exploratory notebook

There is already a time-aware Colab notebook from an earlier stage:

- [sasrec_timeaware_bpi2012_colab_train_01_260512.ipynb](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/sasrec_timeaware_bpi2012_colab_train_01_260512.ipynb)

It was generated from:

- [generate_timeaware_colab_notebook.py](/C:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/scripts/generate_timeaware_colab_notebook.py)

That notebook:

- reused an older baseline candidate (`refine_v3_ml50_do025`)
- compared baseline vs time-aware 8-bucket vs time-aware 9-bucket
- under both `NDCG@10` and `NDCG@5` selection criteria

However, those earlier comparisons were done before the new baseline rerun became the definitive reference.

So the future Codex should **not** blindly reuse that notebook as-is.

It may still be used as a structural template.

---

## What the future Codex should assume

By the time this handoff is used:

- one final baseline run will have already been selected through sanity check
- that selection will likely come from either:
  - `outputs/sasrec_bpi2012_ndcg10`
  - or `outputs/sasrec_bpi2012_ndcg5`

The future Codex must identify:

- the final chosen baseline configuration
- its selection criterion
  - `full_valid_ndcg@10`
  - or `full_valid_ndcg@5`
- the chosen seed set for sanity check

---

## Expected next time-aware experiment design

The user’s intent is:

- use the **one single final baseline run**
- then test time-aware versions of **that same configuration**

This means:

- keep all baseline hyperparameters fixed
- only add time embedding
- vary bucket setup as needed

### Minimum comparison structure

The future notebook will likely need:

1. baseline result reuse
2. time-aware 8-bucket
3. time-aware 9-bucket

Potentially with:

- the same seed(s) used for the final baseline sanity check

### Comparison logic

Baseline should preferably be **reused from existing completed runs**, not retrained again, if:

- the configuration matches exactly
- the selection criterion matches exactly
- the code version is already the fixed one

The notebook should train only the new time-aware runs, then rebuild comparison tables from run folders.

---

## Important evaluation rule to preserve

The user strongly prefers this interpretation rule:

- if best epoch selection is based on `NDCG@10`, then primary performance interpretation should also focus on `NDCG@10`
- if best epoch selection is based on `NDCG@5`, then primary interpretation should focus on `NDCG@5`

Other metrics can still be shown, but the main comparison should follow the selection metric.

This should be preserved in the future time-aware notebook as well.

---

## Important notebook behavior preference

The user explicitly requested that future Colab notebooks should **not** leave `git pull` commented out.

So future notebooks should include active code like:

```python
%cd /content/time-aware-behavior-prediction
!git pull
```

not a commented version.

---

## Recommended structure for the future time-aware notebook

The next Codex should likely create a notebook that:

1. mounts Drive
2. clones repo if needed
3. runs:
   - `%cd /content/time-aware-behavior-prediction`
   - `!git pull`
4. installs requirements for Colab
5. copies processed data from Drive
6. checks that the selected baseline run already exists
7. trains only time-aware runs
8. rebuilds result tables from run folders
9. compares:
   - baseline
   - b8
   - b9
10. summarizes by selection criterion

The structure can be similar to the previous time-aware notebook, but it should be updated to target the **new final baseline**.

---

## What the future Codex must decide when creating the notebook

The next Codex will need to determine:

1. which baseline run was finally selected
2. whether the final baseline comes from:
   - `ndcg10` track
   - or `ndcg5` track
3. whether time-aware should be tested with:
   - one seed first
   - or the same multiple seeds used in sanity check

Recommended default:

- keep the final chosen baseline fixed
- use the same seed(s) that are already considered trustworthy for that final baseline
- compare baseline vs `8-bucket` vs `9-bucket`

---

## Practical warning

Do not mix:

- old pre-fix baseline results
- new post-fix baseline results
- new time-aware results

inside the same interpretation without checking the output directories carefully.

Use direct rebuild from:

- `metrics_summary.json`
- `config.json`

rather than relying only on `experiment_index.csv`.

---

## What the future Codex should deliver

The desired deliverable is a new Colab notebook for time-aware experiments using the final chosen baseline.

That notebook should:

- reuse baseline results if possible,
- train only new time-aware runs,
- support at least:
  - 8-bucket
  - 9-bucket
- and compare them clearly under the proper selection-metric interpretation.

