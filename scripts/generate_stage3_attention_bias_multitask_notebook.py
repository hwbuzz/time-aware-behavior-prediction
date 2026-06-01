from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "sasrec_stage3_bpi2012_colab_train_03_260601.ipynb"


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
            # SASRec Stage 3 Attention Bias Multi-Task BPI2012 Colab Train 03

            This notebook tests the Stage 3 attention-bias multi-task setting on the `anchor_ml20` backbone.

            Main comparison groups:
            - `anchor_single_task`
            - `anchor_attention_bias_single_task`
            - `anchor_multi_task`
            - `anchor_attention_bias_multi_task`

            Main comparison metric:
            - `full ranking + NDCG@10`
            """
        )
    )

    cells.append(
        code_cell(
            """
            import torch

            print('torch version:', torch.__version__)
            print('cuda available:', torch.cuda.is_available())
            if torch.cuda.is_available():
                print('gpu name:', torch.cuda.get_device_name(0))
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

            DATA_DIR = f'{DRIVE_ROOT}/data/processed/bpi2012_complete_only'
            BASELINE_NDCG10_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg10'
            SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_timeaware_attention_bias_ndcg10'
            MULTITASK_BASELINE_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_baseline_multitask_ndcg10'
            MULTITASK_ATTNBIAS_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_attention_bias_multitask_ndcg10'
            NOTEBOOK_DIR = f'{DRIVE_ROOT}/notebooks'

            print('DATA_DIR:', DATA_DIR)
            print('BASELINE_NDCG10_OUTPUT_DIR:', BASELINE_NDCG10_OUTPUT_DIR)
            print('SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR:', SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR)
            print('MULTITASK_BASELINE_OUTPUT_DIR:', MULTITASK_BASELINE_OUTPUT_DIR)
            print('MULTITASK_ATTNBIAS_OUTPUT_DIR:', MULTITASK_ATTNBIAS_OUTPUT_DIR)
            print('NOTEBOOK_DIR:', NOTEBOOK_DIR)
            """
        )
    )

    cells.append(
        code_cell(
            """
            !mkdir -p "$NOTEBOOK_DIR"
            !mkdir -p "$DATA_DIR"
            !mkdir -p "$BASELINE_NDCG10_OUTPUT_DIR"
            !mkdir -p "$SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR"
            !mkdir -p "$MULTITASK_BASELINE_OUTPUT_DIR"
            !mkdir -p "$MULTITASK_ATTNBIAS_OUTPUT_DIR"
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

            skip_packages = ['pywinpty']

            with open('requirements.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()

            with open('requirements_colab.txt', 'w', encoding='utf-8') as f:
                for line in lines:
                    pkg = line.strip().lower()
                    if not any(name in pkg for name in skip_packages):
                        f.write(line)

            print('created requirements_colab.txt')
            """
        )
    )

    cells.append(code_cell("!pip install -r requirements_colab.txt"))

    cells.append(
        markdown_cell(
            """
            ## Verify Stage 3 processed file

            Stage 3 next-time prediction uses the processed time-feature CSV.
            This check confirms that `delta_next_seconds` already exists.
            """
        )
    )

    cells.append(
        code_cell(
            """
            import pandas as pd

            time_features_path = 'data/processed/bpi2012_complete_only/events_encoded_time_features.csv'
            df = pd.read_csv(time_features_path)
            required_cols = [
                'delta_prev_seconds',
                'delta_start_seconds',
                'delta_next_seconds',
            ]
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                raise ValueError(f'Missing required Stage 3 columns: {missing}')

            print('Stage 3 processed file is ready.')
            print(df.columns.tolist())
            df[['user_id', 'event_idx', 'delta_prev_seconds', 'delta_start_seconds', 'delta_next_seconds']].head()
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Experiment design

            Stage 3 attention-bias multi-task runs:

            - backbone: `anchor_ml20`
            - time-aware backbone: `delta_start + 9-bucket attention bias`
            - multi-task outputs: `next activity + next time`
            - time target: `delta_next_seconds`
            - time target transform: `log1p`
            - time loss: `huber`
            - time loss weight: `1.0`
            - best epoch criterion: `full_valid_ndcg@10`
            - final comparison uses all 3 seeds: `42`, `2024`, `7`
            """
        )
    )

    cells.append(markdown_cell("## Check prerequisite reference runs"))

    cells.append(
        code_cell(
            """
            from pathlib import Path

            baseline_required_runs = [
                'anchor_ml20_s42',
                'anchor_ml20_s2024',
                'anchor_ml20_s7',
            ]
            single_task_attnbias_runs = [
                'attnbias_dstart_ml20_b9_s42',
                'attnbias_dstart_ml20_b9_s2024',
                'attnbias_dstart_ml20_b9_s7',
            ]
            multitask_baseline_runs = [
                'multitask_anchor_ml20_s42',
                'multitask_anchor_ml20_s2024',
                'multitask_anchor_ml20_s7',
            ]

            checks = [
                ('baseline', BASELINE_NDCG10_OUTPUT_DIR, baseline_required_runs),
                ('single-task attention bias', SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR, single_task_attnbias_runs),
                ('multi-task baseline', MULTITASK_BASELINE_OUTPUT_DIR, multitask_baseline_runs),
            ]

            print('=' * 80)
            for label, output_dir, run_names in checks:
                print(label)
                base = Path(output_dir)
                for run_name in run_names:
                    run_dir = base / run_name
                    print(' ', run_name, 'EXISTS' if run_dir.exists() else 'MISSING')
                print('-' * 80)
            """
        )
    )

    cells.append(markdown_cell("## Check planned attention-bias multi-task runs"))

    cells.append(
        code_cell(
            """
            planned_attnbias_multitask_runs = [
                'multitask_attnbias_dstart_ml20_b9_s42',
                'multitask_attnbias_dstart_ml20_b9_s2024',
                'multitask_attnbias_dstart_ml20_b9_s7',
            ]

            output_dir = Path(MULTITASK_ATTNBIAS_OUTPUT_DIR)
            print('=' * 80)
            print('Stage 3 attention-bias multi-task runs')
            for run_name in planned_attnbias_multitask_runs:
                run_dir = output_dir / run_name
                print(run_name, 'EXISTS' if run_dir.exists() else 'OK')
            """
        )
    )

    cells.append(markdown_cell("## Train attention-bias multi-task runs"))

    run_templates = [
        ("multitask_attnbias_dstart_ml20_b9_s42", 42),
        ("multitask_attnbias_dstart_ml20_b9_s2024", 2024),
        ("multitask_attnbias_dstart_ml20_b9_s7", 7),
    ]

    for run_name, seed in run_templates:
        cells.append(
            code_cell(
                f"""
                !python src/train_sasrec.py \\
                  --run_name {run_name} \\
                  --hidden_units 32 \\
                  --num_blocks 2 \\
                  --num_heads 1 \\
                  --maxlen 20 \\
                  --lr 0.001 \\
                  --dropout_rate 0.2 \\
                  --seed {seed} \\
                  --use_time_attention_bias \\
                  --time_delta_column delta_start_seconds \\
                  --time_bucket_boundaries 60,600,3600,86400,604800 \\
                  --enable_time_prediction \\
                  --time_prediction_target delta_next_seconds \\
                  --time_target_transform log1p \\
                  --time_loss_type huber \\
                  --time_loss_weight 1.0 \\
                  --output_dir "$MULTITASK_ATTNBIAS_OUTPUT_DIR" \\
                  --selection_metric full_valid_ndcg@10 \\
                  --interactions_path data/processed/bpi2012_complete_only/sasrec_interactions.txt \\
                  --time_features_path data/processed/bpi2012_complete_only/events_encoded_time_features.csv \\
                  --batch_size 128 \\
                  --num_epochs 50 \\
                  --eval_every 5 \\
                  --device cuda \\
                  --num_negative_samples 100 \\
                  --eval_protocol both \\
                  --topk_list 5,10 \\
                  --save_every_eval
                """
            )
        )

    cells.append(
        code_cell(
            """
            from pathlib import Path
            import json
            import pandas as pd

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
                        'run_dir': str(run_dir),
                        'completed_at': summary.get('completed_at'),
                        'best_epoch': summary.get('best_epoch'),
                        'checkpoint_best': summary.get('checkpoint_best'),
                        'checkpoint_last': summary.get('checkpoint_last'),
                        'metrics_history': summary.get('metrics_history'),
                        'config_path': str(config_path),
                        'metrics_summary': str(summary_path),
                        'maxlen': config.get('maxlen'),
                        'dropout_rate': config.get('dropout_rate'),
                        'hidden_units': config.get('hidden_units'),
                        'seed': config.get('seed'),
                        'selection_metric': config.get('selection_metric'),
                        'use_time_attention_bias': config.get('use_time_attention_bias', False),
                        'enable_time_prediction': config.get('enable_time_prediction', False),
                        'time_delta_column': config.get('time_delta_column'),
                        'time_prediction_target': config.get('time_prediction_target'),
                        'time_loss_weight': config.get('time_loss_weight'),
                        'time_target_transform': config.get('time_target_transform'),
                        'time_modeling_mode': config.get('time_modeling_mode'),
                    }
                    for group_name in ['best_valid', 'best_test_at_best_valid', 'last_valid', 'last_test']:
                        group = summary.get(group_name) or {}
                        for mode, metrics in group.items():
                            for key, value in metrics.items():
                                row[f'{group_name}_{mode}_{key}'] = value
                    rows.append(row)
                return pd.DataFrame(rows)
            """
        )
    )

    cells.append(
        code_cell(
            """
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 2000)
            pd.set_option('display.max_colwidth', None)
            """
        )
    )

    cells.append(markdown_cell("## Comparison summary"))

    cells.append(
        code_cell(
            """
            baseline_runs = [
                'anchor_ml20_s42',
                'anchor_ml20_s2024',
                'anchor_ml20_s7',
            ]
            single_task_attnbias_runs = [
                'attnbias_dstart_ml20_b9_s42',
                'attnbias_dstart_ml20_b9_s2024',
                'attnbias_dstart_ml20_b9_s7',
            ]
            multitask_baseline_runs = [
                'multitask_anchor_ml20_s42',
                'multitask_anchor_ml20_s2024',
                'multitask_anchor_ml20_s7',
            ]
            multitask_attnbias_runs = [
                'multitask_attnbias_dstart_ml20_b9_s42',
                'multitask_attnbias_dstart_ml20_b9_s2024',
                'multitask_attnbias_dstart_ml20_b9_s7',
            ]

            baseline_df = rebuild_df(BASELINE_NDCG10_OUTPUT_DIR)
            single_task_attnbias_df = rebuild_df(SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR)
            multitask_baseline_df = rebuild_df(MULTITASK_BASELINE_OUTPUT_DIR)
            multitask_attnbias_df = rebuild_df(MULTITASK_ATTNBIAS_OUTPUT_DIR)

            baseline_subset = baseline_df[baseline_df['run_name'].isin(baseline_runs)].copy()
            baseline_subset['variant'] = 'anchor_single_task'

            single_task_attnbias_subset = single_task_attnbias_df[single_task_attnbias_df['run_name'].isin(single_task_attnbias_runs)].copy()
            single_task_attnbias_subset['variant'] = 'anchor_attnbias_single_task'

            multitask_baseline_subset = multitask_baseline_df[multitask_baseline_df['run_name'].isin(multitask_baseline_runs)].copy()
            multitask_baseline_subset['variant'] = 'anchor_multi_task'

            multitask_attnbias_subset = multitask_attnbias_df[multitask_attnbias_df['run_name'].isin(multitask_attnbias_runs)].copy()
            multitask_attnbias_subset['variant'] = 'anchor_attnbias_multi_task'

            df_compare = pd.concat(
                [
                    baseline_subset,
                    single_task_attnbias_subset,
                    multitask_baseline_subset,
                    multitask_attnbias_subset,
                ],
                ignore_index=True,
            )
            df_compare = df_compare.sort_values(['variant', 'seed', 'run_name']).reset_index(drop=True)

            display_cols = [
                'run_name', 'seed', 'variant', 'maxlen', 'dropout_rate', 'selection_metric', 'best_epoch',
                'best_valid_full_ndcg@5', 'best_valid_full_hr@5',
                'best_valid_full_ndcg@10', 'best_valid_full_hr@10',
                'best_valid_full_mrr',
                'best_test_at_best_valid_full_ndcg@5', 'best_test_at_best_valid_full_hr@5',
                'best_test_at_best_valid_full_ndcg@10', 'best_test_at_best_valid_full_hr@10',
                'best_test_at_best_valid_full_mrr',
                'best_valid_sampled_ndcg@5', 'best_valid_sampled_hr@5',
                'best_valid_sampled_ndcg@10', 'best_valid_sampled_hr@10',
                'best_valid_sampled_mrr',
                'best_test_at_best_valid_sampled_ndcg@5', 'best_test_at_best_valid_sampled_hr@5',
                'best_test_at_best_valid_sampled_ndcg@10', 'best_test_at_best_valid_sampled_hr@10',
                'best_test_at_best_valid_sampled_mrr',
                'best_valid_task_accuracy',
                'best_valid_task_macro_f1',
                'best_valid_task_top5_accuracy',
                'best_valid_task_top10_accuracy',
                'best_valid_task_time_mae',
                'best_valid_task_time_rmse',
                'best_valid_task_time_median_ae',
                'best_test_at_best_valid_task_accuracy',
                'best_test_at_best_valid_task_macro_f1',
                'best_test_at_best_valid_task_top5_accuracy',
                'best_test_at_best_valid_task_top10_accuracy',
                'best_test_at_best_valid_task_time_mae',
                'best_test_at_best_valid_task_time_rmse',
                'best_test_at_best_valid_task_time_median_ae',
            ]

            existing_display_cols = [c for c in display_cols if c in df_compare.columns]
            df_compare[existing_display_cols]
            """
        )
    )

    cells.append(
        code_cell(
            """
            summary_metric_cols = [
                'best_valid_full_ndcg@5', 'best_valid_full_hr@5',
                'best_valid_full_ndcg@10', 'best_valid_full_hr@10',
                'best_valid_full_mrr',
                'best_test_at_best_valid_full_ndcg@5', 'best_test_at_best_valid_full_hr@5',
                'best_test_at_best_valid_full_ndcg@10', 'best_test_at_best_valid_full_hr@10',
                'best_test_at_best_valid_full_mrr',
                'best_valid_sampled_ndcg@5', 'best_valid_sampled_hr@5',
                'best_valid_sampled_ndcg@10', 'best_valid_sampled_hr@10',
                'best_valid_sampled_mrr',
                'best_test_at_best_valid_sampled_ndcg@5', 'best_test_at_best_valid_sampled_hr@5',
                'best_test_at_best_valid_sampled_ndcg@10', 'best_test_at_best_valid_sampled_hr@10',
                'best_test_at_best_valid_sampled_mrr',
                'best_valid_task_accuracy',
                'best_valid_task_macro_f1',
                'best_valid_task_top5_accuracy',
                'best_valid_task_top10_accuracy',
                'best_valid_task_time_mae',
                'best_valid_task_time_rmse',
                'best_valid_task_time_median_ae',
                'best_test_at_best_valid_task_accuracy',
                'best_test_at_best_valid_task_macro_f1',
                'best_test_at_best_valid_task_top5_accuracy',
                'best_test_at_best_valid_task_top10_accuracy',
                'best_test_at_best_valid_task_time_mae',
                'best_test_at_best_valid_task_time_rmse',
                'best_test_at_best_valid_task_time_median_ae',
            ]

            summary_metric_cols = [c for c in summary_metric_cols if c in df_compare.columns]
            summary_compare = df_compare.groupby('variant')[summary_metric_cols].agg(['mean', 'std'])
            summary_compare
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            Interpretation guide:

            - compare `anchor_single_task` vs `anchor_attnbias_single_task`
            - compare `anchor_single_task` vs `anchor_multi_task`
            - compare `anchor_multi_task` vs `anchor_attnbias_multi_task`
            - use `best_test_at_best_valid_full_ndcg@10` as the main Stage 3 comparison metric
            - use mean/std across `42`, `2024`, `7` for final interpretation
            - if attention bias helps in multi-task, the main question is whether it recovers some of the ranking loss seen in the baseline multi-task setting
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
