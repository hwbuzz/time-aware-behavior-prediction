from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "sasrec_stage3_bpi2012_colab_train_08_260607.ipynb"


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
            # SASRec Stage 3 Refine Attention-Bias Multi-Task BPI2012 Colab Train 08

            This notebook extends the Stage 3 attention-bias multi-task experiments to the
            `refine_ml50_do035` backbone.

            Main comparison groups:
            - `refine_single_task`
            - `refine_attnbias_single_task`
            - `refine_multi_task_w1.0`
            - `refine_attnbias_multi_task_w1.0`
            - `refine_attnbias_multi_task_w0.1`

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

            DATA_DIR = f'{DRIVE_ROOT}/data/processed/bpi2012_complete_only_stage3_v2'
            BASELINE_NDCG10_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg10'
            SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_timeaware_attention_bias_ndcg10'
            MULTITASK_BASELINE_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_baseline_multitask_ndcg10_v2'
            MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_refine_attention_bias_multitask_ndcg10_v2'
            MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_stage3_refine_attention_bias_multitask_w01_ndcg10_v2'
            NOTEBOOK_DIR = f'{DRIVE_ROOT}/notebooks'

            print('DATA_DIR:', DATA_DIR)
            print('BASELINE_NDCG10_OUTPUT_DIR:', BASELINE_NDCG10_OUTPUT_DIR)
            print('SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR:', SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR)
            print('MULTITASK_BASELINE_OUTPUT_DIR:', MULTITASK_BASELINE_OUTPUT_DIR)
            print('MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR:', MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR)
            print('MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR:', MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR)
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
            !mkdir -p "$MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR"
            !mkdir -p "$MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR"
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
            ## Prepare Stage 3 processed dataset

            This notebook regenerates the Stage 3 dataset into the versioned Drive folder
            before training, so the run does not depend on any stale local processed files.
            """
        )
    )
    cells.append(
        code_cell(
            """
            %cd /content/time-aware-behavior-prediction
            !python scripts/regenerate_stage3_processed_dataset.py --output-dir "$DATA_DIR" --backup-existing
            !ls "$DATA_DIR"
            """
        )
    )
    cells.append(
        code_cell(
            """
            %cd /content/time-aware-behavior-prediction
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
            import pandas as pd

            time_features_path = 'data/processed/bpi2012_complete_only_stage3_v2/events_encoded_time_features.csv'
            df = pd.read_csv(time_features_path)
            required_cols = ['delta_prev_seconds', 'delta_start_seconds', 'delta_next_seconds']
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

            This notebook tests the `refine_ml50_do035` backbone with:
            - single-task attention bias (`Stage 2` reference)
            - multi-task attention bias with `time_loss_weight=1.0`
            - multi-task attention bias with `time_loss_weight=0.1`

            Fixed settings:
            - backbone: `refine_ml50_do035`
            - time-aware backbone: `delta_start + 9-bucket attention bias`
            - multi-task outputs: `next activity + next time`
            - time target: `delta_next_seconds`
            - time target transform: `log1p`
            - time loss: `huber`
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

            baseline_runs = [
                'refine_ml50_do035_s42',
                'refine_ml50_do035_s2024',
                'refine_ml50_do035_s7',
            ]
            single_task_attnbias_runs = [
                'attnbias_dstart_ml50_do035_b9_s42',
                'attnbias_dstart_ml50_do035_b9_s2024',
                'attnbias_dstart_ml50_do035_b9_s7',
            ]
            multitask_baseline_runs = [
                'multitask_refine_ml50_do035_s42',
                'multitask_refine_ml50_do035_s2024',
                'multitask_refine_ml50_do035_s7',
            ]

            checks = [
                ('baseline', BASELINE_NDCG10_OUTPUT_DIR, baseline_runs),
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

    cells.append(markdown_cell("## Check planned refine attention-bias multi-task runs"))
    cells.append(
        code_cell(
            """
            planned_attnbias_multitask_runs = [
                'multitask_attnbias_dstart_ml50_do035_b9_s42',
                'multitask_attnbias_dstart_ml50_do035_b9_s2024',
                'multitask_attnbias_dstart_ml50_do035_b9_s7',
                'multitask_attnbias_dstart_ml50_do035_b9_w01_s42',
                'multitask_attnbias_dstart_ml50_do035_b9_w01_s2024',
                'multitask_attnbias_dstart_ml50_do035_b9_w01_s7',
            ]

            checks = [
                ('w1.0', MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR),
                ('w0.1', MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR),
            ]

            for label, output_dir in checks:
                base = Path(output_dir)
                print('=' * 80)
                print(f'Stage 3 refine attention-bias multi-task runs ({label})')
                for run_name in planned_attnbias_multitask_runs:
                    if (label == 'w1.0' and '_w01_' in run_name) or (label == 'w0.1' and '_w01_' not in run_name):
                        continue
                    run_dir = base / run_name
                    print(run_name, 'EXISTS' if run_dir.exists() else 'OK')
            """
        )
    )

    cells.append(markdown_cell("## Train refine attention-bias multi-task w1.0 runs"))
    for run_name, seed in [
        ("multitask_attnbias_dstart_ml50_do035_b9_s42", 42),
        ("multitask_attnbias_dstart_ml50_do035_b9_s2024", 2024),
        ("multitask_attnbias_dstart_ml50_do035_b9_s7", 7),
    ]:
        cells.append(
            code_cell(
                f"""
                !python src/train_sasrec.py \\
                  --run_name {run_name} \\
                  --hidden_units 50 \\
                  --num_blocks 2 \\
                  --num_heads 1 \\
                  --maxlen 50 \\
                  --lr 0.001 \\
                  --dropout_rate 0.35 \\
                  --seed {seed} \\
                  --use_time_attention_bias \\
                  --time_delta_column delta_start_seconds \\
                  --time_bucket_boundaries 60,600,3600,86400,604800 \\
                  --enable_time_prediction \\
                  --time_prediction_target delta_next_seconds \\
                  --time_target_transform log1p \\
                  --time_loss_type huber \\
                  --time_loss_weight 1.0 \\
                  --output_dir "$MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR" \\
                  --selection_metric full_valid_ndcg@10 \\
                  --interactions_path data/processed/bpi2012_complete_only_stage3_v2/sasrec_interactions.txt \\
                  --time_features_path data/processed/bpi2012_complete_only_stage3_v2/events_encoded_time_features.csv \\
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

    cells.append(markdown_cell("## Train refine attention-bias multi-task w0.1 runs"))
    for run_name, seed in [
        ("multitask_attnbias_dstart_ml50_do035_b9_w01_s42", 42),
        ("multitask_attnbias_dstart_ml50_do035_b9_w01_s2024", 2024),
        ("multitask_attnbias_dstart_ml50_do035_b9_w01_s7", 7),
    ]:
        cells.append(
            code_cell(
                f"""
                !python src/train_sasrec.py \\
                  --run_name {run_name} \\
                  --hidden_units 50 \\
                  --num_blocks 2 \\
                  --num_heads 1 \\
                  --maxlen 50 \\
                  --lr 0.001 \\
                  --dropout_rate 0.35 \\
                  --seed {seed} \\
                  --use_time_attention_bias \\
                  --time_delta_column delta_start_seconds \\
                  --time_bucket_boundaries 60,600,3600,86400,604800 \\
                  --enable_time_prediction \\
                  --time_prediction_target delta_next_seconds \\
                  --time_target_transform log1p \\
                  --time_loss_type huber \\
                  --time_loss_weight 0.1 \\
                  --output_dir "$MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR" \\
                  --selection_metric full_valid_ndcg@10 \\
                  --interactions_path data/processed/bpi2012_complete_only_stage3_v2/sasrec_interactions.txt \\
                  --time_features_path data/processed/bpi2012_complete_only_stage3_v2/events_encoded_time_features.csv \\
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

    cells.append(markdown_cell("## Load run summaries"))
    cells.append(
        code_cell(
            """
            import json
            from pathlib import Path
            import pandas as pd

            def rebuild_df(output_dir):
                rows = []
                for run_dir in sorted(Path(output_dir).iterdir()):
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
                        metrics = summary.get(group_name, {})
                        for metric_name, value in metrics.items():
                            row[f'{group_name}_{metric_name}'] = value
                    rows.append(row)
                return pd.DataFrame(rows)
            """
        )
    )

    cells.append(markdown_cell("## Compare refine single-task / plain multi-task / attention-bias multi-task"))
    cells.append(
        code_cell(
            """
            baseline_runs = [
                'refine_ml50_do035_s42',
                'refine_ml50_do035_s2024',
                'refine_ml50_do035_s7',
            ]
            single_task_attnbias_runs = [
                'attnbias_dstart_ml50_do035_b9_s42',
                'attnbias_dstart_ml50_do035_b9_s2024',
                'attnbias_dstart_ml50_do035_b9_s7',
            ]
            multitask_baseline_runs = [
                'multitask_refine_ml50_do035_s42',
                'multitask_refine_ml50_do035_s2024',
                'multitask_refine_ml50_do035_s7',
            ]
            multitask_attnbias_runs = [
                'multitask_attnbias_dstart_ml50_do035_b9_s42',
                'multitask_attnbias_dstart_ml50_do035_b9_s2024',
                'multitask_attnbias_dstart_ml50_do035_b9_s7',
            ]
            multitask_attnbias_w01_runs = [
                'multitask_attnbias_dstart_ml50_do035_b9_w01_s42',
                'multitask_attnbias_dstart_ml50_do035_b9_w01_s2024',
                'multitask_attnbias_dstart_ml50_do035_b9_w01_s7',
            ]

            baseline_df = rebuild_df(BASELINE_NDCG10_OUTPUT_DIR)
            single_task_attnbias_df = rebuild_df(SINGLE_TASK_ATTNBIAS_NDCG10_OUTPUT_DIR)
            multitask_baseline_df = rebuild_df(MULTITASK_BASELINE_OUTPUT_DIR)
            multitask_attnbias_df = rebuild_df(MULTITASK_REFINE_ATTNBIAS_OUTPUT_DIR)
            multitask_attnbias_w01_df = rebuild_df(MULTITASK_REFINE_ATTNBIAS_W01_OUTPUT_DIR)

            baseline_subset = baseline_df[baseline_df['run_name'].isin(baseline_runs)].copy()
            baseline_subset['variant'] = 'refine_single_task'

            single_task_attnbias_subset = single_task_attnbias_df[single_task_attnbias_df['run_name'].isin(single_task_attnbias_runs)].copy()
            single_task_attnbias_subset['variant'] = 'refine_attnbias_single_task'

            multitask_baseline_subset = multitask_baseline_df[multitask_baseline_df['run_name'].isin(multitask_baseline_runs)].copy()
            multitask_baseline_subset['variant'] = 'refine_multi_task_w1.0'

            multitask_attnbias_subset = multitask_attnbias_df[multitask_attnbias_df['run_name'].isin(multitask_attnbias_runs)].copy()
            multitask_attnbias_subset['variant'] = 'refine_attnbias_multi_task_w1.0'

            multitask_attnbias_w01_subset = multitask_attnbias_w01_df[multitask_attnbias_w01_df['run_name'].isin(multitask_attnbias_w01_runs)].copy()
            multitask_attnbias_w01_subset['variant'] = 'refine_attnbias_multi_task_w0.1'

            df_compare = pd.concat(
                [
                    baseline_subset,
                    single_task_attnbias_subset,
                    multitask_baseline_subset,
                    multitask_attnbias_subset,
                    multitask_attnbias_w01_subset,
                ],
                ignore_index=True,
            )
            df_compare = df_compare.sort_values(['variant', 'seed', 'run_name']).reset_index(drop=True)

            id_cols = [
                'run_name', 'seed', 'variant', 'maxlen', 'dropout_rate',
                'selection_metric', 'time_loss_weight', 'best_epoch',
            ]

            metric_prefixes = (
                'best_valid_',
                'best_test_at_best_valid_',
                'last_valid_',
                'last_test_',
            )
            metric_cols = [c for c in df_compare.columns if c.startswith(metric_prefixes)]
            display_cols = [c for c in id_cols if c in df_compare.columns] + sorted(metric_cols)

            df_compare[display_cols]
            """
        )
    )

    cells.append(
        code_cell(
            """
            summary_metric_cols = [c for c in df_compare.columns if c.startswith(('best_valid_', 'best_test_at_best_valid_'))]
            summary_compare = df_compare.groupby('variant')[summary_metric_cols].agg(['mean', 'std'])
            summary_compare
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## What to look at

            Read the results in this order:
            - compare `refine_single_task` vs `refine_attnbias_single_task`
            - compare `refine_multi_task_w1.0` vs `refine_attnbias_multi_task_w1.0`
            - compare `refine_attnbias_multi_task_w1.0` vs `refine_attnbias_multi_task_w0.1`
            - then decide whether refine behaves like anchor or shows a different trade-off pattern
            """
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[ok] wrote notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
