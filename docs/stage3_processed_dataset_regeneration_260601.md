Stage 3 processed dataset regeneration

Why this exists

- Stage 1 and Stage 2 used the Drive processed dataset created on April 20, 2026.
- That dataset is internally consistent and was verified against the corrected preprocessing code for the columns used in Stage 2:
  - `sasrec_interactions`
  - `timestamp`
  - `delta_prev_seconds`
  - `delta_start_seconds`
- Stage 3 additionally needs `delta_next_seconds`.
- A later local processed dataset generated on June 1, 2026 had corrupted `events_*` files due to mixed timestamp parsing and CSV roundtrip issues.

Safety policy

- Do not overwrite the Stage 2 dataset folder:
  - `data/processed/bpi2012_complete_only`
- Regenerate Stage 3 into a separate versioned folder:
  - `data/processed/bpi2012_complete_only_stage3_v2`
- Do not reuse old Stage 3 output directories.
- Use versioned Stage 3 output dirs:
  - `outputs/sasrec_stage3_baseline_multitask_ndcg10_v2`
  - `outputs/sasrec_stage3_attention_bias_multitask_ndcg10_v2`

What changed in preprocessing

- Mixed timestamp strings are now normalized safely with `format="mixed"`.
- `event_idx` is normalized to numeric before sorting and interaction building.
- A validation step checks that:
  - required Stage 3 columns exist
  - no unexpected missing values remain
  - `sasrec_interactions` matches `encoded_events`

Files involved

- Preprocessing code:
  - `src/preprocess_bpi2012.py`
- Stage 3 dataset regeneration script:
  - `scripts/regenerate_stage3_processed_dataset.py`
- Stage 3 notebooks using the new dataset:
  - `notebooks/sasrec_stage3_bpi2012_colab_train_02_260601.ipynb`
  - `notebooks/sasrec_stage3_bpi2012_colab_train_03_260601.ipynb`

Recommended regeneration command

From the repo root:

`python scripts/regenerate_stage3_processed_dataset.py --backup-existing`

Default behavior

- input:
  - `data/interim/bpi2012_events_complete_only.csv`
- output:
  - `data/processed/bpi2012_complete_only_stage3_v2`

What the regeneration script writes

- `events_complete_only_filtered.csv`
- `events_encoded_time_features.csv`
- `user_map.csv`
- `item_map.csv`
- `sasrec_interactions.csv`
- `sasrec_interactions.txt`
- `stage3_dataset_metadata.json`

What to check after regeneration

- `stage3_dataset_metadata.json`
- blank counts for:
  - `timestamp`
  - `delta_prev_seconds`
  - `delta_start_seconds`
  - `delta_next_seconds`
- all should be `0`

How Stage 3 notebooks now use data

- The notebooks no longer rely on whatever happens to exist in local `data/processed/...`.
- They copy the Drive folder
  - `data/processed/bpi2012_complete_only_stage3_v2`
  into the Colab repo workspace before training.
- This matches the safer Stage 2 pattern.

Interpretation policy

- Stage 1 results: usable
- Stage 2 results: usable
- Old Stage 3 results from the corrupted local processed dataset: do not rely on them
- New Stage 3 results should be produced only from `bpi2012_complete_only_stage3_v2`
