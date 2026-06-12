from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "sasrec_timeaware_bpi2012_colab_train_09_260612.ipynb"


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


OUTPUT_ROOT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs"
BASELINE_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_bpi2012_ndcg10"
ANCHOR_BUCKET_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_anchor_ml20_ndcg10"
REFINE_BUCKET_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_refine_ml50_do035_ndcg10"
REFINE_DSTART_BUCKET_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_deltastart_refine_ml50_do035_b9_ndcg10"
REFINE_DSTART_CONT_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_deltastart_continuous_refine_ml50_do035_ndcg10"
ANCHOR_ATTN_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_attention_bias_ndcg10"
REFINE_ATTN_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_attention_bias_refine_ndcg10"
SINUSOIDAL_NDCG10_OUTPUT = f"{OUTPUT_ROOT}/sasrec_timeaware_sinusoidal_ndcg10"


SEEDS = [42, 2024, 7]

ANCHOR_BASELINE_RUNS = [f"anchor_ml20_s{seed}" for seed in SEEDS]
REFINE_BASELINE_RUNS = [f"refine_ml50_do035_s{seed}" for seed in SEEDS]

ANCHOR_BUCKET_RUNS = [f"timeaware_anchor_ml20_b9_s{seed}" for seed in SEEDS]
REFINE_BUCKET_RUNS = [f"timeaware_refine_ml50_do035_b9_s{seed}" for seed in SEEDS]
REFINE_DSTART_BUCKET_RUNS = [f"timeaware_dstart_refine_ml50_do035_b9_s{seed}" for seed in SEEDS]
REFINE_DSTART_CONT_RUNS = [f"timeaware_dstart_conti_refine_ml50_do035_s{seed}" for seed in SEEDS]

ANCHOR_ATTN_RUNS = [f"attnbias_dstart_ml20_b9_s{seed}" for seed in SEEDS]
REFINE_ATTN_RUNS = [f"attnbias_dstart_ml50_do035_b9_s{seed}" for seed in SEEDS]

ANCHOR_SIN_DPREV_RUNS = [f"timeaware_dprev_sinusoidal_ml20_s{seed}" for seed in SEEDS]
ANCHOR_SIN_DSTART_RUNS = [f"timeaware_dstart_sinusoidal_ml20_s{seed}" for seed in SEEDS]
REFINE_SIN_DPREV_RUNS = [f"timeaware_dprev_sinusoidal_ml50_do035_s{seed}" for seed in SEEDS]
REFINE_SIN_DSTART_RUNS = [f"timeaware_dstart_sinusoidal_ml50_do035_s{seed}" for seed in SEEDS]


def train_cell(run_name: str, maxlen: int, dropout_rate: float, seed: int, time_delta_column: str) -> str:
    return dedent(
        f"""
        !python src/train_sasrec.py \\
          --run_name {run_name} \\
          --hidden_units 50 \\
          --num_blocks 2 \\
          --num_heads 1 \\
          --maxlen {maxlen} \\
          --lr 0.001 \\
          --dropout_rate {dropout_rate} \\
          --seed {seed} \\
          --use_time_embedding \\
          --time_features_path data/processed/bpi2012_complete_only/events_encoded_time_features.csv \\
          --time_delta_column {time_delta_column} \\
          --time_encoding sinusoidal \\
          --time_sinusoidal_base 10000 \\
          --output_dir "$SINUSOIDAL_NDCG10_OUTPUT_DIR" \\
          --selection_metric full_valid_ndcg@10 \\
          --interactions_path data/processed/bpi2012_complete_only/sasrec_interactions.txt \\
          --batch_size 128 \\
          --num_epochs 50 \\
          --eval_every 5 \\
          --device cuda \\
          --num_negative_samples 100 \\
          --eval_protocol both \\
          --topk_list 5,10 \\
          --save_every_eval
        """
    ).strip()


def append_train_section(cells, title: str, run_names: list[str], maxlen: int, dropout_rate: float, time_delta_column: str) -> None:
    cells.append(markdown_cell(f"## Train `{title}`"))
    for run_name, seed in zip(run_names, SEEDS):
        cells.append(markdown_cell(f"### `{run_name}`"))
        cells.append(code_cell(train_cell(run_name, maxlen, dropout_rate, seed, time_delta_column)))


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        markdown_cell(
            """
            # SASRec Time-Aware BPI2012 Colab Train 09 (`Additive_sinusoidal`)

            Stage 2 follow-up notebook for the new additive time-aware variant:
            `Additive_sinusoidal`.

            Goals:
            - train `Additive_sinusoidal` with both `delta_prev_seconds` and `delta_start_seconds`
            - use both backbones: `anchor_ml20`, `refine_ml50_do035`
            - keep the usual 3 seeds: `42`, `2024`, `7`
            - compare with completed Stage 2 runs, including `attention bias`
            - display not only ranking metrics, but also task metrics such as `accuracy`, `macro_f1`, `top5_accuracy`, `top10_accuracy` when available
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
    cells.append(code_cell("from google.colab import drive\ndrive.mount('/content/drive')"))
    cells.append(
        code_cell(
            f"""
            GITHUB_USERNAME = 'hwbuzz'

            DRIVE_ROOT = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction'
            REPO_DIR = '/content/time-aware-behavior-prediction'

            DATA_DIR = f'{{DRIVE_ROOT}}/data/processed/bpi2012_complete_only'
            OUTPUT_ROOT = '{OUTPUT_ROOT}'
            BASELINE_NDCG10_OUTPUT_DIR = '{BASELINE_NDCG10_OUTPUT}'
            ANCHOR_BUCKET_NDCG10_OUTPUT_DIR = '{ANCHOR_BUCKET_NDCG10_OUTPUT}'
            REFINE_BUCKET_NDCG10_OUTPUT_DIR = '{REFINE_BUCKET_NDCG10_OUTPUT}'
            REFINE_DSTART_BUCKET_NDCG10_OUTPUT_DIR = '{REFINE_DSTART_BUCKET_NDCG10_OUTPUT}'
            REFINE_DSTART_CONT_NDCG10_OUTPUT_DIR = '{REFINE_DSTART_CONT_NDCG10_OUTPUT}'
            ANCHOR_ATTN_NDCG10_OUTPUT_DIR = '{ANCHOR_ATTN_NDCG10_OUTPUT}'
            REFINE_ATTN_NDCG10_OUTPUT_DIR = '{REFINE_ATTN_NDCG10_OUTPUT}'
            SINUSOIDAL_NDCG10_OUTPUT_DIR = '{SINUSOIDAL_NDCG10_OUTPUT}'
            NOTEBOOK_DIR = f'{{DRIVE_ROOT}}/notebooks'

            print('DATA_DIR:', DATA_DIR)
            print('SINUSOIDAL_NDCG10_OUTPUT_DIR:', SINUSOIDAL_NDCG10_OUTPUT_DIR)
            """
        )
    )
    cells.append(
        code_cell(
            """
            !mkdir -p "$NOTEBOOK_DIR"
            !mkdir -p "$DATA_DIR"
            !mkdir -p "$SINUSOIDAL_NDCG10_OUTPUT_DIR"
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
    cells.append(code_cell('!ls "$DATA_DIR"'))
    cells.append(
        code_cell(
            """
            %cd /content/time-aware-behavior-prediction
            !mkdir -p data/processed
            !rm -rf data/processed/bpi2012_complete_only
            !cp -r "$DATA_DIR" data/processed/
            !ls data/processed/bpi2012_complete_only
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Experiment design

            New runs to train:

            - `anchor_ml20 + Additive_sinusoidal + delta_prev_seconds`
            - `anchor_ml20 + Additive_sinusoidal + delta_start_seconds`
            - `refine_ml50_do035 + Additive_sinusoidal + delta_prev_seconds`
            - `refine_ml50_do035 + Additive_sinusoidal + delta_start_seconds`

            Common setting:

            - seeds: `42`, `2024`, `7`
            - selection metric: `full_valid_ndcg@10`
            - eval protocol: `both`
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            from pathlib import Path

            comparison_run_groups = {{
                'anchor_baseline': {ANCHOR_BASELINE_RUNS!r},
                'anchor_bucket_b9': {ANCHOR_BUCKET_RUNS!r},
                'anchor_attnbias_dstart_b9': {ANCHOR_ATTN_RUNS!r},
                'anchor_sinusoidal_dprev': {ANCHOR_SIN_DPREV_RUNS!r},
                'anchor_sinusoidal_dstart': {ANCHOR_SIN_DSTART_RUNS!r},
                'refine_baseline': {REFINE_BASELINE_RUNS!r},
                'refine_bucket_b9': {REFINE_BUCKET_RUNS!r},
                'refine_dstart_bucket_b9': {REFINE_DSTART_BUCKET_RUNS!r},
                'refine_dstart_continuous': {REFINE_DSTART_CONT_RUNS!r},
                'refine_attnbias_dstart_b9': {REFINE_ATTN_RUNS!r},
                'refine_sinusoidal_dprev': {REFINE_SIN_DPREV_RUNS!r},
                'refine_sinusoidal_dstart': {REFINE_SIN_DSTART_RUNS!r},
            }}

            scan_dirs = [
                Path(BASELINE_NDCG10_OUTPUT_DIR),
                Path(ANCHOR_BUCKET_NDCG10_OUTPUT_DIR),
                Path(REFINE_BUCKET_NDCG10_OUTPUT_DIR),
                Path(REFINE_DSTART_BUCKET_NDCG10_OUTPUT_DIR),
                Path(REFINE_DSTART_CONT_NDCG10_OUTPUT_DIR),
                Path(ANCHOR_ATTN_NDCG10_OUTPUT_DIR),
                Path(REFINE_ATTN_NDCG10_OUTPUT_DIR),
                Path(SINUSOIDAL_NDCG10_OUTPUT_DIR),
            ]

            print('=' * 80)
            print('Existing comparison runs')
            for group_name, run_names in comparison_run_groups.items():
                print('-' * 80)
                print(group_name)
                for run_name in run_names:
                    found = any((output_dir / run_name).exists() for output_dir in scan_dirs)
                    print(run_name, 'EXISTS' if found else 'MISSING')
            """
        )
    )

    append_train_section(
        cells,
        "anchor_ml20 + Additive_sinusoidal + delta_prev_seconds",
        ANCHOR_SIN_DPREV_RUNS,
        20,
        0.2,
        "delta_prev_seconds",
    )
    append_train_section(
        cells,
        "anchor_ml20 + Additive_sinusoidal + delta_start_seconds",
        ANCHOR_SIN_DSTART_RUNS,
        20,
        0.2,
        "delta_start_seconds",
    )
    append_train_section(
        cells,
        "refine_ml50_do035 + Additive_sinusoidal + delta_prev_seconds",
        REFINE_SIN_DPREV_RUNS,
        50,
        0.35,
        "delta_prev_seconds",
    )
    append_train_section(
        cells,
        "refine_ml50_do035 + Additive_sinusoidal + delta_start_seconds",
        REFINE_SIN_DSTART_RUNS,
        50,
        0.35,
        "delta_start_seconds",
    )

    cells.append(
        markdown_cell(
            """
            ## Rebuild result tables

            This notebook reads result folders directly from `metrics_summary.json`.
            Older Stage 2 runs may not contain all task metrics; in that case the corresponding cells appear as `NaN`.
            """
        )
    )
    cells.append(
        code_cell(
            """
            from pathlib import Path
            import json
            import pandas as pd

            def rebuild_df_from_dirs(output_dirs):
                rows = []
                seen_runs = set()
                for output_dir in output_dirs:
                    output_path = Path(output_dir)
                    if not output_path.exists():
                        continue
                    for run_dir in output_path.iterdir():
                        if not run_dir.is_dir() or run_dir.name in seen_runs:
                            continue
                        summary_path = run_dir / 'metrics_summary.json'
                        config_path = run_dir / 'config.json'
                        if not summary_path.exists() or not config_path.exists():
                            continue
                        summary = json.loads(summary_path.read_text(encoding='utf-8'))
                        config = json.loads(config_path.read_text(encoding='utf-8'))
                        row = {
                            'run_name': summary.get('run_name'),
                            'source_output_dir': str(output_path),
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
                            'use_time_embedding': config.get('use_time_embedding', False),
                            'use_time_attention_bias': config.get('use_time_attention_bias', False),
                            'time_modeling_mode': config.get('time_modeling_mode'),
                            'time_encoding': config.get('time_encoding'),
                            'time_delta_column': config.get('time_delta_column'),
                            'time_bucket_boundaries_parsed': config.get('time_bucket_boundaries_parsed'),
                            'time_attention_bias_bucket_count': config.get('time_attention_bias_bucket_count'),
                            'time_sinusoidal_base': config.get('time_sinusoidal_base'),
                        }
                        for group_name in ['best_valid', 'best_test_at_best_valid', 'last_valid', 'last_test']:
                            group = summary.get(group_name) or {}
                            for mode, metrics in group.items():
                                for key, value in metrics.items():
                                    row[f'{group_name}_{mode}_{key}'] = value
                        rows.append(row)
                        seen_runs.add(run_dir.name)
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
            f"""
            scan_dirs = [
                BASELINE_NDCG10_OUTPUT_DIR,
                ANCHOR_BUCKET_NDCG10_OUTPUT_DIR,
                REFINE_BUCKET_NDCG10_OUTPUT_DIR,
                REFINE_DSTART_BUCKET_NDCG10_OUTPUT_DIR,
                REFINE_DSTART_CONT_NDCG10_OUTPUT_DIR,
                ANCHOR_ATTN_NDCG10_OUTPUT_DIR,
                REFINE_ATTN_NDCG10_OUTPUT_DIR,
                SINUSOIDAL_NDCG10_OUTPUT_DIR,
            ]

            run_to_variant = {{}}
            for run_name in {ANCHOR_BASELINE_RUNS!r}:
                run_to_variant[run_name] = 'anchor_baseline'
            for run_name in {ANCHOR_BUCKET_RUNS!r}:
                run_to_variant[run_name] = 'anchor_bucket_b9'
            for run_name in {ANCHOR_ATTN_RUNS!r}:
                run_to_variant[run_name] = 'anchor_attnbias_dstart_b9'
            for run_name in {ANCHOR_SIN_DPREV_RUNS!r}:
                run_to_variant[run_name] = 'anchor_sinusoidal_dprev'
            for run_name in {ANCHOR_SIN_DSTART_RUNS!r}:
                run_to_variant[run_name] = 'anchor_sinusoidal_dstart'

            for run_name in {REFINE_BASELINE_RUNS!r}:
                run_to_variant[run_name] = 'refine_baseline'
            for run_name in {REFINE_BUCKET_RUNS!r}:
                run_to_variant[run_name] = 'refine_bucket_b9'
            for run_name in {REFINE_DSTART_BUCKET_RUNS!r}:
                run_to_variant[run_name] = 'refine_dstart_bucket_b9'
            for run_name in {REFINE_DSTART_CONT_RUNS!r}:
                run_to_variant[run_name] = 'refine_dstart_continuous'
            for run_name in {REFINE_ATTN_RUNS!r}:
                run_to_variant[run_name] = 'refine_attnbias_dstart_b9'
            for run_name in {REFINE_SIN_DPREV_RUNS!r}:
                run_to_variant[run_name] = 'refine_sinusoidal_dprev'
            for run_name in {REFINE_SIN_DSTART_RUNS!r}:
                run_to_variant[run_name] = 'refine_sinusoidal_dstart'

            df_all = rebuild_df_from_dirs(scan_dirs)
            df_compare = df_all[df_all['run_name'].isin(run_to_variant)].copy()
            df_compare['variant'] = df_compare['run_name'].map(run_to_variant)
            df_compare = df_compare.sort_values(['variant', 'seed', 'run_name']).reset_index(drop=True)

            display_cols = [
                'run_name', 'seed', 'variant', 'maxlen', 'dropout_rate', 'selection_metric',
                'time_modeling_mode', 'time_encoding', 'time_delta_column', 'time_attention_bias_bucket_count', 'time_sinusoidal_base',
                'best_valid_full_ndcg@10', 'best_valid_full_hr@10', 'best_valid_full_ndcg@5', 'best_valid_full_hr@5', 'best_valid_full_mrr',
                'best_test_at_best_valid_full_ndcg@10', 'best_test_at_best_valid_full_hr@10', 'best_test_at_best_valid_full_ndcg@5', 'best_test_at_best_valid_full_hr@5', 'best_test_at_best_valid_full_mrr',
                'best_valid_sampled_ndcg@10', 'best_valid_sampled_hr@10', 'best_valid_sampled_ndcg@5', 'best_valid_sampled_hr@5', 'best_valid_sampled_mrr',
                'best_test_at_best_valid_sampled_ndcg@10', 'best_test_at_best_valid_sampled_hr@10', 'best_test_at_best_valid_sampled_ndcg@5', 'best_test_at_best_valid_sampled_hr@5', 'best_test_at_best_valid_sampled_mrr',
                'best_valid_task_accuracy', 'best_valid_task_macro_f1', 'best_valid_task_top5_accuracy', 'best_valid_task_top10_accuracy',
                'best_test_at_best_valid_task_accuracy', 'best_test_at_best_valid_task_macro_f1', 'best_test_at_best_valid_task_top5_accuracy', 'best_test_at_best_valid_task_top10_accuracy',
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
                'best_valid_full_ndcg@10', 'best_valid_full_hr@10', 'best_valid_full_ndcg@5', 'best_valid_full_hr@5', 'best_valid_full_mrr',
                'best_test_at_best_valid_full_ndcg@10', 'best_test_at_best_valid_full_hr@10', 'best_test_at_best_valid_full_ndcg@5', 'best_test_at_best_valid_full_hr@5', 'best_test_at_best_valid_full_mrr',
                'best_valid_sampled_ndcg@10', 'best_valid_sampled_hr@10', 'best_valid_sampled_ndcg@5', 'best_valid_sampled_hr@5', 'best_valid_sampled_mrr',
                'best_test_at_best_valid_sampled_ndcg@10', 'best_test_at_best_valid_sampled_hr@10', 'best_test_at_best_valid_sampled_ndcg@5', 'best_test_at_best_valid_sampled_hr@5', 'best_test_at_best_valid_sampled_mrr',
                'best_valid_task_accuracy', 'best_valid_task_macro_f1', 'best_valid_task_top5_accuracy', 'best_valid_task_top10_accuracy',
                'best_test_at_best_valid_task_accuracy', 'best_test_at_best_valid_task_macro_f1', 'best_test_at_best_valid_task_top5_accuracy', 'best_test_at_best_valid_task_top10_accuracy',
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

            - main comparison metric: `best_test_at_best_valid_full_ndcg@10`
            - compare `Additive_sinusoidal` against completed baseline / additive / attention-bias runs
            - task metrics (`accuracy`, `macro_f1`, `top5_accuracy`, `top10_accuracy`) are shown when the corresponding run folders already contain them
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
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
