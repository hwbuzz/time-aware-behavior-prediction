from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "sasrec_stage3_bpi2012_time_naive_baselines_07_260602.ipynb"


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        markdown_cell(
            """
            # Stage 3 Time Naive Baselines BPI2012 Colab 07

            This notebook computes simple next-time naive baselines for Stage 3 using the
            same leave-one-out split logic as the SASRec experiments.

            Naive baselines:
            - global mean
            - global median
            - current activity mean
            - prefix length mean

            Evaluation metrics:
            - MAE
            - RMSE
            - Median AE
            """
        )
    )

    cells.append(
        code_cell(
            """
            from google.colab import drive
            drive.mount('/content/drive')
            """
        )
    )

    cells.append(
        code_cell(
            """
            GITHUB_USERNAME = 'hwbuzz'

            DRIVE_ROOT = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction'
            REPO_DIR = '/content/time-aware-behavior-prediction'

            DATA_DIR = f'{DRIVE_ROOT}/data/processed/bpi2012_complete_only_stage3_v2'
            MULTITASK_BASELINE_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_baseline_multitask_ndcg10_v2'
            MULTITASK_BASELINE_W01_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_baseline_multitask_w01_ndcg10_v2'
            MULTITASK_ATTNBIAS_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_attention_bias_multitask_ndcg10_v2'
            MULTITASK_ATTNBIAS_W01_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_attention_bias_multitask_w01_ndcg10_v2'

            print('DATA_DIR:', DATA_DIR)
            """
        )
    )

    cells.append(
        code_cell(
            """
            %cd /content
            !test -d time-aware-behavior-prediction || git clone https://github.com/$GITHUB_USERNAME/time-aware-behavior-prediction.git
            %cd /content/time-aware-behavior-prediction
            !git pull
            """
        )
    )

    cells.append(
        code_cell(
            """
            %cd /content/time-aware-behavior-prediction
            !python scripts/regenerate_stage3_processed_dataset.py --output-dir "$DATA_DIR" --backup-existing
            !mkdir -p data/processed
            !rm -rf data/processed/bpi2012_complete_only_stage3_v2
            !cp -r "$DATA_DIR" data/processed/
            !ls data/processed/bpi2012_complete_only_stage3_v2
            """
        )
    )

    cells.append(
        code_cell(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Build train / valid / test rows with the same split logic

            The SASRec split is leave-one-out:
            - train = sequence except last 2 events
            - valid target = second last event
            - test target = last event

            For next-time:
            - train samples use every current event in train with its `delta_next_seconds`
            - valid sample uses the last train event -> valid event gap
            - test sample uses the valid event -> test event gap
            """
        )
    )

    cells.append(
        code_cell(
            """
            interactions_path = 'data/processed/bpi2012_complete_only_stage3_v2/sasrec_interactions.txt'
            time_features_path = 'data/processed/bpi2012_complete_only_stage3_v2/events_encoded_time_features.csv'

            interactions = pd.read_csv(
                interactions_path,
                sep=' ',
                header=None,
                names=['user_id', 'item_id'],
            )

            timef = pd.read_csv(time_features_path)
            timef['user_id'] = pd.to_numeric(timef['user_id']).astype(int)
            timef['event_idx'] = pd.to_numeric(timef['event_idx']).astype(int)
            timef['item_id'] = pd.to_numeric(timef['item_id']).astype(int)
            timef['delta_next_seconds'] = pd.to_numeric(timef['delta_next_seconds'], errors='coerce')

            interactions['user_id'] = interactions['user_id'].astype(int)
            interactions['item_id'] = interactions['item_id'].astype(int)

            # Sanity check alignment with processed file order
            merged = interactions.copy()
            merged['event_idx'] = merged.groupby('user_id').cumcount()
            check = merged.merge(
                timef[['user_id', 'event_idx', 'item_id', 'activity', 'delta_next_seconds']],
                on=['user_id', 'event_idx', 'item_id'],
                how='left',
            )
            if check['activity'].isna().any():
                raise ValueError('Failed to align interactions with time feature rows.')

            def build_split_rows(df):
                train_rows = []
                valid_rows = []
                test_rows = []

                for user_id, g in df.groupby('user_id', sort=True):
                    g = g.sort_values('event_idx').reset_index(drop=True)
                    n = len(g)
                    if n < 4:
                        continue

                    train_g = g.iloc[:-2].reset_index(drop=True)
                    valid_g = g.iloc[[-2]].reset_index(drop=True)
                    test_g = g.iloc[[-1]].reset_index(drop=True)

                    # train rows: each current event in train predicts its next gap
                    for i, row in train_g.iterrows():
                        train_rows.append({
                            'split': 'train',
                            'user_id': user_id,
                            'event_idx': int(row['event_idx']),
                            'current_activity': row['activity'],
                            'prefix_length': int(i + 1),
                            'y_true': float(row['delta_next_seconds']),
                        })

                    # valid row: current event is last train event
                    last_train = train_g.iloc[-1]
                    valid_rows.append({
                        'split': 'valid',
                        'user_id': user_id,
                        'event_idx': int(last_train['event_idx']),
                        'current_activity': last_train['activity'],
                        'prefix_length': int(len(train_g)),
                        'y_true': float(last_train['delta_next_seconds']),
                    })

                    # test row: current event is valid event (second last original)
                    valid_event = valid_g.iloc[0]
                    test_rows.append({
                        'split': 'test',
                        'user_id': user_id,
                        'event_idx': int(valid_event['event_idx']),
                        'current_activity': valid_event['activity'],
                        'prefix_length': int(len(train_g) + 1),
                        'y_true': float(valid_event['delta_next_seconds']),
                    })

                return (
                    pd.DataFrame(train_rows),
                    pd.DataFrame(valid_rows),
                    pd.DataFrame(test_rows),
                )

            train_df, valid_df, test_df = build_split_rows(check)

            print('train rows:', len(train_df))
            print('valid rows:', len(valid_df))
            print('test rows:', len(test_df))

            train_df.head()
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Compute naive baselines

            Baselines:
            - `global_mean`
            - `global_median`
            - `activity_mean`
            - `prefix_len_mean`
            """
        )
    )

    cells.append(
        code_cell(
            """
            def mae(y_true, y_pred):
                y_true = np.asarray(y_true, dtype=float)
                y_pred = np.asarray(y_pred, dtype=float)
                return float(np.mean(np.abs(y_true - y_pred)))

            def rmse(y_true, y_pred):
                y_true = np.asarray(y_true, dtype=float)
                y_pred = np.asarray(y_pred, dtype=float)
                return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            def median_ae(y_true, y_pred):
                y_true = np.asarray(y_true, dtype=float)
                y_pred = np.asarray(y_pred, dtype=float)
                return float(np.median(np.abs(y_true - y_pred)))

            global_mean = float(train_df['y_true'].mean())
            global_median = float(train_df['y_true'].median())
            activity_mean_map = train_df.groupby('current_activity')['y_true'].mean().to_dict()
            prefix_mean_map = train_df.groupby('prefix_length')['y_true'].mean().to_dict()

            def predict_global_mean(df):
                return np.full(len(df), global_mean, dtype=float)

            def predict_global_median(df):
                return np.full(len(df), global_median, dtype=float)

            def predict_activity_mean(df):
                fallback = global_mean
                return np.array([activity_mean_map.get(act, fallback) for act in df['current_activity']], dtype=float)

            def predict_prefix_mean(df):
                fallback = global_mean
                return np.array([prefix_mean_map.get(int(pl), fallback) for pl in df['prefix_length']], dtype=float)

            baseline_predictors = {
                'global_mean': predict_global_mean,
                'global_median': predict_global_median,
                'activity_mean': predict_activity_mean,
                'prefix_len_mean': predict_prefix_mean,
            }

            rows = []
            for split_name, split_df in [('valid', valid_df), ('test', test_df)]:
                for baseline_name, predictor in baseline_predictors.items():
                    pred = predictor(split_df)
                    rows.append({
                        'split': split_name,
                        'baseline': baseline_name,
                        'mae': mae(split_df['y_true'], pred),
                        'rmse': rmse(split_df['y_true'], pred),
                        'median_ae': median_ae(split_df['y_true'], pred),
                    })

            baseline_results = pd.DataFrame(rows)
            baseline_results
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Convert seconds to readable units
            """
        )
    )

    cells.append(
        code_cell(
            """
            readable = baseline_results.copy()
            for col in ['mae', 'rmse', 'median_ae']:
                readable[f'{col}_hours'] = readable[col] / 3600.0
                readable[f'{col}_minutes'] = readable[col] / 60.0

            readable
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Compare with Stage 3 model results

            This block extracts saved Stage 3 multitask results and compares their
            test next-time metrics with the naive baselines.
            """
        )
    )

    cells.append(
        code_cell(
            """
            def rebuild_df(output_dir: str):
                rows = []
                output_path = Path(output_dir)
                if not output_path.exists():
                    return pd.DataFrame()
                for run_dir in output_path.iterdir():
                    if not run_dir.is_dir():
                        continue
                    summary_path = run_dir / 'metrics_summary.json'
                    config_path = run_dir / 'config.json'
                    if not summary_path.exists() or not config_path.exists():
                        continue
                    summary = json.loads(summary_path.read_text(encoding='utf-8'))
                    config = json.loads(config_path.read_text(encoding='utf-8'))
                    row = {
                        'run_name': summary.get('run_name'),
                        'seed': config.get('seed'),
                        'time_loss_weight': config.get('time_loss_weight'),
                    }
                    for group_name in ['best_test_at_best_valid']:
                        group = summary.get(group_name) or {}
                        for mode, metrics in group.items():
                            for key, value in metrics.items():
                                row[f'{group_name}_{mode}_{key}'] = value
                    rows.append(row)
                return pd.DataFrame(rows)

            mt_w10_df = rebuild_df(MULTITASK_BASELINE_OUTPUT_DIR)
            mt_anchor_w01_df = rebuild_df(MULTITASK_BASELINE_W01_OUTPUT_DIR)
            mt_attnbias_w10_df = rebuild_df(MULTITASK_ATTNBIAS_OUTPUT_DIR)
            mt_attnbias_w01_df = rebuild_df(MULTITASK_ATTNBIAS_W01_OUTPUT_DIR)

            model_rows = []

            for _, r in mt_w10_df[mt_w10_df['run_name'].isin([
                'multitask_anchor_ml20_s42',
                'multitask_anchor_ml20_s2024',
                'multitask_anchor_ml20_s7',
            ])].iterrows():
                model_rows.append({
                    'model': 'anchor_multi_task_w1.0',
                    'mae': r.get('best_test_at_best_valid_task_time_mae'),
                    'rmse': r.get('best_test_at_best_valid_task_time_rmse'),
                    'median_ae': r.get('best_test_at_best_valid_task_time_median_ae'),
                })

            for _, r in mt_anchor_w01_df[mt_anchor_w01_df['run_name'].isin([
                'multitask_anchor_ml20_w01_s42',
                'multitask_anchor_ml20_w01_s2024',
                'multitask_anchor_ml20_w01_s7',
            ])].iterrows():
                model_rows.append({
                    'model': 'anchor_multi_task_w0.1',
                    'mae': r.get('best_test_at_best_valid_task_time_mae'),
                    'rmse': r.get('best_test_at_best_valid_task_time_rmse'),
                    'median_ae': r.get('best_test_at_best_valid_task_time_median_ae'),
                })

            for _, r in mt_attnbias_w10_df[mt_attnbias_w10_df['run_name'].isin([
                'multitask_attnbias_dstart_ml20_b9_s42',
                'multitask_attnbias_dstart_ml20_b9_s2024',
                'multitask_attnbias_dstart_ml20_b9_s7',
            ])].iterrows():
                model_rows.append({
                    'model': 'anchor_attnbias_multi_task_w1.0',
                    'mae': r.get('best_test_at_best_valid_task_time_mae'),
                    'rmse': r.get('best_test_at_best_valid_task_time_rmse'),
                    'median_ae': r.get('best_test_at_best_valid_task_time_median_ae'),
                })

            for _, r in mt_attnbias_w01_df[mt_attnbias_w01_df['run_name'].isin([
                'multitask_attnbias_dstart_ml20_b9_w01_s42',
                'multitask_attnbias_dstart_ml20_b9_w01_s2024',
                'multitask_attnbias_dstart_ml20_b9_w01_s7',
            ])].iterrows():
                model_rows.append({
                    'model': 'anchor_attnbias_multi_task_w0.1',
                    'mae': r.get('best_test_at_best_valid_task_time_mae'),
                    'rmse': r.get('best_test_at_best_valid_task_time_rmse'),
                    'median_ae': r.get('best_test_at_best_valid_task_time_median_ae'),
                })

            model_df = pd.DataFrame(model_rows)
            model_summary = model_df.groupby('model')[['mae', 'rmse', 'median_ae']].agg(['mean', 'std'])
            model_summary
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            Interpretation guide:

            - if a multitask model beats `global_mean` / `global_median`, it is at least better than trivial prediction
            - if it also beats `activity_mean` and `prefix_len_mean`, the next-time head is learning something beyond simple heuristics
            - `median_ae` reflects typical-case error
            - `mae` and `rmse` reflect tail sensitivity
            """
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.x"}

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
