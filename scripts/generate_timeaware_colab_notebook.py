import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


COMMON_PREFIX = """--interactions_path data/processed/bpi2012_complete_only/sasrec_interactions.txt \\
  --batch_size 128 \\
  --num_epochs 50 \\
  --eval_every 5 \\
  --device cuda \\
  --num_negative_samples 100 \\
  --eval_protocol both \\
  --topk_list 5,10 \\
  --save_every_eval"""


BASELINE_NDCG10_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012"
BASELINE_NDCG5_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg5"
TIMEAWARE_NDCG10_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_timeaware_bpi2012"
TIMEAWARE_NDCG5_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_timeaware_bpi2012_ndcg5"


BASELINE_NDCG10_RUNS = [
    "refine_v3_ml50_do025_seed42",
    "refine_v3_ml50_do025_seed2024",
]

BASELINE_NDCG5_RUNS = [
    "refine_v3_ml50_do025_seed42_ndcg5",
    "refine_v3_ml50_do025_seed2024_ndcg5",
]


TIMEAWARE_PARAMS = dict(
    hidden_units=50,
    num_blocks=2,
    num_heads=1,
    maxlen=50,
    lr=0.001,
    dropout_rate=0.25,
)


NDCG10_NEW_RUNS = [
    (
        "timeaware01_refine_v3_ml50_do025_seed42_b8",
        dict(seed=42, boundaries="60,600,3600,86400"),
    ),
    (
        "timeaware01_refine_v3_ml50_do025_seed2024_b8",
        dict(seed=2024, boundaries="60,600,3600,86400"),
    ),
    (
        "timeaware01_refine_v3_ml50_do025_seed42_b9",
        dict(seed=42, boundaries="60,600,3600,86400,604800"),
    ),
    (
        "timeaware01_refine_v3_ml50_do025_seed2024_b9",
        dict(seed=2024, boundaries="60,600,3600,86400,604800"),
    ),
]


NDCG5_NEW_RUNS = [
    (
        "timeaware01_refine_v3_ml50_do025_seed42_b8_ndcg5",
        dict(seed=42, boundaries="60,600,3600,86400"),
    ),
    (
        "timeaware01_refine_v3_ml50_do025_seed2024_b8_ndcg5",
        dict(seed=2024, boundaries="60,600,3600,86400"),
    ),
    (
        "timeaware01_refine_v3_ml50_do025_seed42_b9_ndcg5",
        dict(seed=42, boundaries="60,600,3600,86400,604800"),
    ),
    (
        "timeaware01_refine_v3_ml50_do025_seed2024_b9_ndcg5",
        dict(seed=2024, boundaries="60,600,3600,86400,604800"),
    ),
]


def train_cell(run_name: str, output_dir: str, selection_metric: str, params: dict) -> str:
    return "\n".join(
        [
            "!python src/train_sasrec.py \\",
            f"  --run_name {run_name} \\",
            f"  --hidden_units {TIMEAWARE_PARAMS['hidden_units']} \\",
            f"  --num_blocks {TIMEAWARE_PARAMS['num_blocks']} \\",
            f"  --num_heads {TIMEAWARE_PARAMS['num_heads']} \\",
            f"  --maxlen {TIMEAWARE_PARAMS['maxlen']} \\",
            f"  --lr {TIMEAWARE_PARAMS['lr']} \\",
            f"  --dropout_rate {TIMEAWARE_PARAMS['dropout_rate']} \\",
            f"  --seed {params['seed']} \\",
            "  --use_time_embedding \\",
            "  --time_delta_column delta_prev_seconds \\",
            "  --time_bucket_first_event_separate \\",
            "  --time_bucket_zero_gap_separate \\",
            f"  --time_bucket_boundaries {params['boundaries']} \\",
            f'  --output_dir "{output_dir}" \\',
            f"  --selection_metric {selection_metric} \\",
            f"  {COMMON_PREFIX}",
            "",
        ]
    )


def rebuild_df_code() -> str:
    return (
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n\n"
        "def rebuild_df(output_dir: str):\n"
        "    rows = []\n"
        "    output_path = Path(output_dir)\n"
        "    if not output_path.exists():\n"
        "        return pd.DataFrame()\n"
        "    for run_dir in output_path.iterdir():\n"
        "        if not run_dir.is_dir():\n"
        "            continue\n"
        "        summary_path = run_dir / 'metrics_summary.json'\n"
        "        config_path = run_dir / 'config.json'\n"
        "        if not summary_path.exists() or not config_path.exists():\n"
        "            continue\n"
        "        summary = json.loads(summary_path.read_text(encoding='utf-8'))\n"
        "        config = json.loads(config_path.read_text(encoding='utf-8'))\n"
        "        row = {\n"
        "            'run_name': summary.get('run_name'),\n"
        "            'run_dir': str(run_dir),\n"
        "            'completed_at': summary.get('completed_at'),\n"
        "            'best_epoch': summary.get('best_epoch'),\n"
        "            'checkpoint_best': summary.get('checkpoint_best'),\n"
        "            'checkpoint_last': summary.get('checkpoint_last'),\n"
        "            'metrics_history': summary.get('metrics_history'),\n"
        "            'config_path': str(config_path),\n"
        "            'metrics_summary': str(summary_path),\n"
        "            'maxlen': config.get('maxlen'),\n"
        "            'dropout_rate': config.get('dropout_rate'),\n"
        "            'hidden_units': config.get('hidden_units'),\n"
        "            'seed': config.get('seed'),\n"
        "            'selection_metric': config.get('selection_metric'),\n"
        "            'use_time_embedding': config.get('use_time_embedding', False),\n"
        "            'time_bucket_boundaries': ','.join(str(x) for x in config.get('time_bucket_boundaries', [])),\n"
        "            'time_bucket_count': config.get('time_bucket_count'),\n"
        "        }\n"
        "        best_valid = summary.get('best_valid', {})\n"
        "        best_test = summary.get('best_test_at_best_valid', {})\n"
        "        def pick(metrics_group, mode, key):\n"
        "            return metrics_group.get(mode, {}).get(key)\n"
        "        row.update({\n"
        "            'best_valid_full_ndcg@10': pick(best_valid, 'full', 'ndcg@10'),\n"
        "            'best_valid_full_hr@10': pick(best_valid, 'full', 'hr@10'),\n"
        "            'best_valid_full_ndcg@5': pick(best_valid, 'full', 'ndcg@5'),\n"
        "            'best_valid_full_hr@5': pick(best_valid, 'full', 'hr@5'),\n"
        "            'best_valid_full_mrr': pick(best_valid, 'full', 'mrr'),\n"
        "            'best_test_full_ndcg@10': pick(best_test, 'full', 'ndcg@10'),\n"
        "            'best_test_full_hr@10': pick(best_test, 'full', 'hr@10'),\n"
        "            'best_test_full_ndcg@5': pick(best_test, 'full', 'ndcg@5'),\n"
        "            'best_test_full_hr@5': pick(best_test, 'full', 'hr@5'),\n"
        "            'best_test_full_mrr': pick(best_test, 'full', 'mrr'),\n"
        "            'best_valid_sampled_ndcg@10': pick(best_valid, 'sampled', 'ndcg@10'),\n"
        "            'best_valid_sampled_hr@10': pick(best_valid, 'sampled', 'hr@10'),\n"
        "            'best_valid_sampled_ndcg@5': pick(best_valid, 'sampled', 'ndcg@5'),\n"
        "            'best_valid_sampled_hr@5': pick(best_valid, 'sampled', 'hr@5'),\n"
        "            'best_valid_sampled_mrr': pick(best_valid, 'sampled', 'mrr'),\n"
        "            'best_test_sampled_ndcg@10': pick(best_test, 'sampled', 'ndcg@10'),\n"
        "            'best_test_sampled_hr@10': pick(best_test, 'sampled', 'hr@10'),\n"
        "            'best_test_sampled_ndcg@5': pick(best_test, 'sampled', 'ndcg@5'),\n"
        "            'best_test_sampled_hr@5': pick(best_test, 'sampled', 'hr@5'),\n"
        "            'best_test_sampled_mrr': pick(best_test, 'sampled', 'mrr'),\n"
        "        })\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows)\n"
    )


def comparison_cell(
    label: str,
    baseline_output_dir_var: str,
    timeaware_output_dir_var: str,
    baseline_runs: list[str],
    timeaware_runs: list[str],
    primary_valid_col: str,
    primary_test_col: str,
) -> list[dict]:
    baseline_list_name = f"{label.lower()}_baseline_runs".replace("@", "").replace(" ", "_")
    timeaware_list_name = f"{label.lower()}_timeaware_runs".replace("@", "").replace(" ", "_")
    df_name = f"df_{label.lower().replace('@', '').replace(' ', '_')}"
    summary_name = f"summary_{label.lower().replace('@', '').replace(' ', '_')}"
    return [
        md_cell(f"## {label} comparison summary\n"),
        code_cell(
            f"{baseline_list_name} = [\n" +
            "".join([f"    '{name}',\n" for name in baseline_runs]) +
            "]\n" +
            f"{timeaware_list_name} = [\n" +
            "".join([f"    '{name}',\n" for name in timeaware_runs]) +
            "]\n\n" +
            f"baseline_df = rebuild_df({baseline_output_dir_var})\n" +
            f"timeaware_df = rebuild_df({timeaware_output_dir_var})\n\n" +
            f"baseline_subset = baseline_df[baseline_df['run_name'].isin({baseline_list_name})].copy()\n" +
            f"baseline_subset['model_variant'] = 'baseline'\n" +
            f"baseline_subset['bucket_variant'] = 'baseline'\n\n" +
            f"timeaware_subset = timeaware_df[timeaware_df['run_name'].isin({timeaware_list_name})].copy()\n" +
            f"timeaware_subset['model_variant'] = 'timeaware'\n" +
            "timeaware_subset['bucket_variant'] = timeaware_subset['run_name'].apply(lambda x: 'b8' if '_b8' in x else 'b9')\n\n" +
            f"{df_name} = pd.concat([baseline_subset, timeaware_subset], ignore_index=True)\n" +
            f"{df_name} = {df_name}.sort_values(['bucket_variant', 'seed', 'run_name']).reset_index(drop=True)\n" +
            f"{df_name}[[\n"
            "    'run_name', 'seed', 'bucket_variant', 'use_time_embedding', 'time_bucket_boundaries',\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]]\n"
        ),
        code_cell(
            f"{summary_name} = {df_name}.groupby('bucket_variant')[[\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]].agg(['mean', 'std'])\n"
            f"{summary_name}\n"
        ),
        md_cell(
            f"Interpretation guide for {label}:\n"
            f"- first compare `{primary_valid_col}` and `{primary_test_col}` across `baseline`, `b8`, and `b9`\n"
            "- then check whether sampled metrics and MRR show a similar trend\n"
            "- because baseline is reused from existing runs, only the time-aware runs are newly trained here\n"
        ),
    ]


cells = [
    md_cell(
        "# SASRec Time-Aware BPI2012 Colab Train 01\n\n"
        "Colab notebook for the first time-aware SASRec experiment on BPI 2012.\n\n"
        "Goals:\n"
        "- reuse the existing `refine_v3_ml50_do025` baseline results instead of retraining them\n"
        "- train only the new time-aware runs with time-delta bucket embeddings\n"
        "- compare baseline vs `8-bucket` vs `9-bucket` under both `NDCG@10` and `NDCG@5` model-selection criteria\n"
    ),
    code_cell(
        "import torch\n\n"
        "print('torch version:', torch.__version__)\n"
        "print('cuda available:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print('gpu name:', torch.cuda.get_device_name(0))\n"
    ),
    code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
    ),
    code_cell(
        "GITHUB_USERNAME = 'hwbuzz'\n\n"
        "DRIVE_ROOT = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction'\n"
        "REPO_DIR = '/content/time-aware-behavior-prediction'\n\n"
        "DATA_DIR = f'{DRIVE_ROOT}/data/processed/bpi2012_complete_only'\n"
        f"BASELINE_NDCG10_OUTPUT_DIR = '{BASELINE_NDCG10_OUTPUT}'\n"
        f"BASELINE_NDCG5_OUTPUT_DIR = '{BASELINE_NDCG5_OUTPUT}'\n"
        f"TIMEAWARE_NDCG10_OUTPUT_DIR = '{TIMEAWARE_NDCG10_OUTPUT}'\n"
        f"TIMEAWARE_NDCG5_OUTPUT_DIR = '{TIMEAWARE_NDCG5_OUTPUT}'\n"
        "NOTEBOOK_DIR = f'{DRIVE_ROOT}/notebooks'\n\n"
        "print('DATA_DIR:', DATA_DIR)\n"
        "print('BASELINE_NDCG10_OUTPUT_DIR:', BASELINE_NDCG10_OUTPUT_DIR)\n"
        "print('BASELINE_NDCG5_OUTPUT_DIR:', BASELINE_NDCG5_OUTPUT_DIR)\n"
        "print('TIMEAWARE_NDCG10_OUTPUT_DIR:', TIMEAWARE_NDCG10_OUTPUT_DIR)\n"
        "print('TIMEAWARE_NDCG5_OUTPUT_DIR:', TIMEAWARE_NDCG5_OUTPUT_DIR)\n"
    ),
    code_cell(
        "!mkdir -p \"$NOTEBOOK_DIR\"\n"
        "!mkdir -p \"$DATA_DIR\"\n"
        "!mkdir -p \"$BASELINE_NDCG10_OUTPUT_DIR\"\n"
        "!mkdir -p \"$BASELINE_NDCG5_OUTPUT_DIR\"\n"
        "!mkdir -p \"$TIMEAWARE_NDCG10_OUTPUT_DIR\"\n"
        "!mkdir -p \"$TIMEAWARE_NDCG5_OUTPUT_DIR\"\n"
    ),
    code_cell(
        "%cd /content\n"
        "!test -d time-aware-behavior-prediction || git clone https://github.com/$GITHUB_USERNAME/time-aware-behavior-prediction.git\n"
        "%cd /content/time-aware-behavior-prediction\n"
    ),
    code_cell(
        "# If you need the latest code from GitHub, uncomment below.\n"
        "# %cd /content/time-aware-behavior-prediction\n"
        "# !git pull\n"
    ),
    code_cell(
        "%cd /content/time-aware-behavior-prediction\n\n"
        "skip_packages = ['pywinpty']\n\n"
        "with open('requirements.txt', 'r', encoding='utf-8') as f:\n"
        "    lines = f.readlines()\n\n"
        "with open('requirements_colab.txt', 'w', encoding='utf-8') as f:\n"
        "    for line in lines:\n"
        "        pkg = line.strip().lower()\n"
        "        if not any(name in pkg for name in skip_packages):\n"
        "            f.write(line)\n\n"
        "print('created requirements_colab.txt')\n"
    ),
    code_cell("!pip install -r requirements_colab.txt\n"),
    code_cell('!ls "$DATA_DIR"\n'),
    code_cell(
        "%cd /content/time-aware-behavior-prediction\n"
        "!mkdir -p data/processed\n"
        "!cp -r \"$DATA_DIR\" data/processed/\n"
        "!ls data/processed/bpi2012_complete_only\n"
    ),
    md_cell(
        "## Experiment design\n\n"
        "Fixed baseline setting:\n"
        "- `refine_v3_ml50_do025`\n"
        "- `hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.25`\n"
        "- seeds: `42`, `2024`\n\n"
        "Comparison targets:\n"
        "- baseline (reuse existing completed runs)\n"
        "- time-aware `8-bucket`\n"
        "- time-aware `9-bucket`\n\n"
        "Bucket designs:\n"
        "- `8-bucket`: `padding`, `first`, `zero-gap`, `(0,1m)`, `[1m,10m)`, `[10m,1h)`, `[1h,1d)`, `[>=1d]`\n"
        "- `9-bucket`: `padding`, `first`, `zero-gap`, `(0,1m)`, `[1m,10m)`, `[10m,1h)`, `[1h,1d)`, `[1d,7d)`, `[>=7d]`\n"
    ),
    md_cell(
        "## Check existing baseline runs\n\n"
        "These baseline runs should already exist and must not be retrained.\n"
    ),
    code_cell(
        "from pathlib import Path\n\n"
        "baseline_ndcg10_runs = [\n" +
        "".join([f"    '{name}',\n" for name in BASELINE_NDCG10_RUNS]) +
        "]\n"
        "baseline_ndcg5_runs = [\n" +
        "".join([f"    '{name}',\n" for name in BASELINE_NDCG5_RUNS]) +
        "]\n\n"
        "for label, output_dir, run_names in [\n"
        "    ('Baseline NDCG@10', Path(BASELINE_NDCG10_OUTPUT_DIR), baseline_ndcg10_runs),\n"
        "    ('Baseline NDCG@5', Path(BASELINE_NDCG5_OUTPUT_DIR), baseline_ndcg5_runs),\n"
        "]:\n"
        "    print('=' * 80)\n"
        "    print(label)\n"
        "    for run_name in run_names:\n"
        "        run_dir = output_dir / run_name\n"
        "        print(run_name, 'EXISTS' if run_dir.exists() else 'MISSING')\n"
    ),
    md_cell(
        "## Check planned time-aware runs\n\n"
        "Only train runs that are still missing.\n"
    ),
    code_cell(
        "planned_ndcg10 = [\n" +
        "".join([f"    '{run_name}',\n" for run_name, _ in NDCG10_NEW_RUNS]) +
        "]\n"
        "planned_ndcg5 = [\n" +
        "".join([f"    '{run_name}',\n" for run_name, _ in NDCG5_NEW_RUNS]) +
        "]\n\n"
        "for label, output_dir, run_names in [\n"
        "    ('Time-aware NDCG@10', Path(TIMEAWARE_NDCG10_OUTPUT_DIR), planned_ndcg10),\n"
        "    ('Time-aware NDCG@5', Path(TIMEAWARE_NDCG5_OUTPUT_DIR), planned_ndcg5),\n"
        "]:\n"
        "    print('=' * 80)\n"
        "    print(label)\n"
        "    for run_name in run_names:\n"
        "        run_dir = output_dir / run_name\n"
        "        print(run_name, 'EXISTS' if run_dir.exists() else 'OK')\n"
    ),
    md_cell(
        "## Train time-aware runs for `NDCG@10`\n\n"
        "Run these cells only if the corresponding run directory does not already exist.\n"
    ),
]

for run_name, params in NDCG10_NEW_RUNS:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(train_cell(run_name, TIMEAWARE_NDCG10_OUTPUT, "full_valid_ndcg@10", params)))

cells.append(
    md_cell(
        "## Train time-aware runs for `NDCG@5`\n\n"
        "Run these cells only if the corresponding run directory does not already exist.\n"
    )
)

for run_name, params in NDCG5_NEW_RUNS:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(train_cell(run_name, TIMEAWARE_NDCG5_OUTPUT, "full_valid_ndcg@5", params)))

cells.extend(
    [
        md_cell(
            "## Rebuild result tables from run folders\n\n"
            "This avoids schema issues in `experiment_index.csv` and lets us combine old baseline runs with new time-aware runs safely.\n"
        ),
        code_cell(rebuild_df_code()),
    ]
)

cells.extend(
    comparison_cell(
        "NDCG@10",
        "BASELINE_NDCG10_OUTPUT_DIR",
        "TIMEAWARE_NDCG10_OUTPUT_DIR",
        BASELINE_NDCG10_RUNS,
        [name for name, _ in NDCG10_NEW_RUNS],
        "best_valid_full_ndcg@10",
        "best_test_full_ndcg@10",
    )
)

cells.extend(
    comparison_cell(
        "NDCG@5",
        "BASELINE_NDCG5_OUTPUT_DIR",
        "TIMEAWARE_NDCG5_OUTPUT_DIR",
        BASELINE_NDCG5_RUNS,
        [name for name, _ in NDCG5_NEW_RUNS],
        "best_valid_full_ndcg@5",
        "best_test_full_ndcg@5",
    )
)


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "collapsed_sections": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebooks") / "sasrec_timeaware_bpi2012_colab_train_01_260512.ipynb"
output_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(output_path)
