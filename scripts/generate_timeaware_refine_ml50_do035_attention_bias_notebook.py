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


BASELINE_NDCG10_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg10"
BASELINE_NDCG5_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg5"
TIMEAWARE_NDCG10_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_timeaware_attention_bias_ndcg10"
TIMEAWARE_NDCG5_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_timeaware_attention_bias_ndcg5"


BASELINE_RUNS = [
    "refine_ml50_do035_s42",
    "refine_ml50_do035_s2024",
    "refine_ml50_do035_s7",
]


BASELINE_PARAMS = dict(
    hidden_units=50,
    num_blocks=2,
    num_heads=1,
    maxlen=50,
    lr=0.001,
    dropout_rate=0.35,
)


NDCG10_RUNS = [
    ("attnbias_dstart_ml50_do035_b9_s42", dict(seed=42)),
    ("attnbias_dstart_ml50_do035_b9_s2024", dict(seed=2024)),
    ("attnbias_dstart_ml50_do035_b9_s7", dict(seed=7)),
]

NDCG5_RUNS = [
    ("attnbias_dstart_ml50_do035_b9_s42", dict(seed=42)),
    ("attnbias_dstart_ml50_do035_b9_s2024", dict(seed=2024)),
    ("attnbias_dstart_ml50_do035_b9_s7", dict(seed=7)),
]


def train_cell(run_name: str, output_dir: str, selection_metric: str, seed: int) -> str:
    return "\n".join(
        [
            "!python src/train_sasrec.py \\",
            f"  --run_name {run_name} \\",
            f"  --hidden_units {BASELINE_PARAMS['hidden_units']} \\",
            f"  --num_blocks {BASELINE_PARAMS['num_blocks']} \\",
            f"  --num_heads {BASELINE_PARAMS['num_heads']} \\",
            f"  --maxlen {BASELINE_PARAMS['maxlen']} \\",
            f"  --lr {BASELINE_PARAMS['lr']} \\",
            f"  --dropout_rate {BASELINE_PARAMS['dropout_rate']} \\",
            f"  --seed {seed} \\",
            "  --use_time_attention_bias \\",
            "  --time_delta_column delta_start_seconds \\",
            "  --time_bucket_boundaries 60,600,3600,86400,604800 \\",
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
        "            'use_time_attention_bias': config.get('use_time_attention_bias', False),\n"
        "            'time_modeling_mode': config.get('time_modeling_mode'),\n"
        "            'time_encoding': config.get('time_encoding'),\n"
        "            'time_delta_column': config.get('time_delta_column'),\n"
        "            'time_bucket_boundaries_parsed': config.get('time_bucket_boundaries_parsed'),\n"
        "            'time_attention_bias_bucket_count': config.get('time_attention_bias_bucket_count'),\n"
        "            'primary_metric_name': config.get('selection_metric'),\n"
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


def comparison_section(
    label: str,
    baseline_output_dir_var: str,
    timeaware_output_dir_var: str,
    baseline_runs: list[str],
    timeaware_runs: list[str],
    primary_valid_col: str,
    primary_test_col: str,
) -> list[dict]:
    df_name = f"df_{label.lower().replace('@', '').replace(' ', '_')}"
    summary_name = f"summary_{label.lower().replace('@', '').replace(' ', '_')}"
    return [
        md_cell(f"## {label} comparison summary\n"),
        code_cell(
            "baseline_runs = [\n"
            + "".join([f"    '{name}',\n" for name in baseline_runs])
            + "]\n"
            + "timeaware_runs = [\n"
            + "".join([f"    '{name}',\n" for name in timeaware_runs])
            + "]\n\n"
            + f"baseline_df = rebuild_df({baseline_output_dir_var})\n"
            + f"timeaware_df = rebuild_df({timeaware_output_dir_var})\n\n"
            + "baseline_subset = baseline_df[baseline_df['run_name'].isin(baseline_runs)].copy()\n"
            + "baseline_subset['model_variant'] = 'baseline'\n"
            + "baseline_subset['time_variant'] = 'baseline'\n\n"
            + "timeaware_subset = timeaware_df[timeaware_df['run_name'].isin(timeaware_runs)].copy()\n"
            + "timeaware_subset['model_variant'] = 'timeaware'\n"
            + "timeaware_subset['time_variant'] = 'attention_bias_dstart_b9'\n\n"
            + f"{df_name} = pd.concat([baseline_subset, timeaware_subset], ignore_index=True)\n"
            + f"{df_name} = {df_name}.sort_values(['time_variant', 'seed', 'run_name']).reset_index(drop=True)\n"
            + f"{df_name}[[\n"
            + "    'run_name', 'seed', 'time_variant', 'use_time_embedding', 'use_time_attention_bias', 'time_modeling_mode',\n"
            + "    'time_delta_column', 'time_bucket_boundaries_parsed', 'time_attention_bias_bucket_count',\n"
            + "    'best_valid_full_ndcg@10', 'best_valid_full_hr@10',\n"
            + "    'best_valid_full_ndcg@5', 'best_valid_full_hr@5',\n"
            + "    'best_valid_full_mrr',\n"
            + "    'best_test_full_ndcg@10', 'best_test_full_hr@10',\n"
            + "    'best_test_full_ndcg@5', 'best_test_full_hr@5',\n"
            + "    'best_test_full_mrr',\n"
            + "    'best_valid_sampled_ndcg@10', 'best_valid_sampled_hr@10',\n"
            + "    'best_valid_sampled_ndcg@5', 'best_valid_sampled_hr@5',\n"
            + "    'best_valid_sampled_mrr',\n"
            + "    'best_test_sampled_ndcg@10', 'best_test_sampled_hr@10',\n"
            + "    'best_test_sampled_ndcg@5', 'best_test_sampled_hr@5',\n"
            + "    'best_test_sampled_mrr',\n"
            + "]]\n"
        ),
        code_cell(
            f"{summary_name} = {df_name}.groupby('time_variant')[[\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]].agg(['mean', 'std'])\n"
            f"{summary_name}\n"
        ),
        md_cell(
            f"Interpretation guide for {label}:\n"
            f"- compare `{primary_valid_col}` and `{primary_test_col}` first\n"
            "- then check whether sampled and MRR move in the same direction\n"
            "- baseline is reused; only attention-bias runs are newly trained here\n"
        ),
    ]


cells = [
    md_cell(
        "# SASRec Attention-Bias Time-Aware BPI2012 Colab Train (`refine_ml50_do035` baseline)\n\n"
        "Colab notebook for the first attention-bias time-aware experiment on top of `refine_ml50_do035`.\n\n"
        "Design:\n"
        "- baseline reuse: `refine_ml50_do035`\n"
        "- time source: `delta_start_seconds`\n"
        "- pairwise causal gap attention bias\n"
        "- 9-bucket scalar attention bias\n"
        "- evaluate under both `NDCG@10` and `NDCG@5` model-selection criteria\n"
    ),
    code_cell(
        "import torch\n\n"
        "print('torch version:', torch.__version__)\n"
        "print('cuda available:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print('gpu name:', torch.cuda.get_device_name(0))\n"
    ),
    code_cell("from google.colab import drive\ndrive.mount('/content/drive')\n"),
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
        "!git pull\n"
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
        "- `refine_ml50_do035`\n"
        "- `hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout=0.35`\n"
        "- seeds: `42`, `2024`, `7`\n\n"
        "Time-aware design:\n"
        "- `delta_start_seconds`\n"
        "- causal pairwise gap attention bias\n"
        "- 9-bucket scalar bias\n"
        "- no additive time embedding in this notebook\n"
    ),
    md_cell("## Check existing baseline runs\n"),
    code_cell(
        "from pathlib import Path\n\n"
        "baseline_runs = [\n"
        + "".join([f"    '{name}',\n" for name in BASELINE_RUNS])
        + "]\n\n"
        "for label, output_dir in [\n"
        "    ('Baseline NDCG@10', Path(BASELINE_NDCG10_OUTPUT_DIR)),\n"
        "    ('Baseline NDCG@5', Path(BASELINE_NDCG5_OUTPUT_DIR)),\n"
        "]:\n"
        "    print('=' * 80)\n"
        "    print(label)\n"
        "    for run_name in baseline_runs:\n"
        "        run_dir = output_dir / run_name\n"
        "        print(run_name, 'EXISTS' if run_dir.exists() else 'MISSING')\n"
    ),
    md_cell("## Check planned new runs\n"),
    code_cell(
        "planned_ndcg10 = [\n"
        + "".join([f"    '{name}',\n" for name, _ in NDCG10_RUNS])
        + "]\n"
        "planned_ndcg5 = [\n"
        + "".join([f"    '{name}',\n" for name, _ in NDCG5_RUNS])
        + "]\n\n"
        "for label, output_dir, run_names in [\n"
        "    ('Attention-Bias NDCG@10', Path(TIMEAWARE_NDCG10_OUTPUT_DIR), planned_ndcg10),\n"
        "    ('Attention-Bias NDCG@5', Path(TIMEAWARE_NDCG5_OUTPUT_DIR), planned_ndcg5),\n"
        "]:\n"
        "    print('=' * 80)\n"
        "    print(label)\n"
        "    for run_name in run_names:\n"
        "        run_dir = output_dir / run_name\n"
        "        print(run_name, 'EXISTS' if run_dir.exists() else 'OK')\n"
    ),
    md_cell("## Train attention-bias runs for `NDCG@10`\n"),
]


for run_name, params in NDCG10_RUNS:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(train_cell(run_name, TIMEAWARE_NDCG10_OUTPUT, "full_valid_ndcg@10", params["seed"])))


cells.append(md_cell("## Train attention-bias runs for `NDCG@5`\n"))


for run_name, params in NDCG5_RUNS:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(train_cell(run_name, TIMEAWARE_NDCG5_OUTPUT, "full_valid_ndcg@5", params["seed"])))


cells.extend(
    [
        md_cell("## Rebuild result tables\n"),
        code_cell(rebuild_df_code()),
    ]
)


cells.extend(
    comparison_section(
        "NDCG@10",
        "BASELINE_NDCG10_OUTPUT_DIR",
        "TIMEAWARE_NDCG10_OUTPUT_DIR",
        BASELINE_RUNS,
        [name for name, _ in NDCG10_RUNS],
        "best_valid_full_ndcg@10",
        "best_test_full_ndcg@10",
    )
)


cells.extend(
    comparison_section(
        "NDCG@5",
        "BASELINE_NDCG5_OUTPUT_DIR",
        "TIMEAWARE_NDCG5_OUTPUT_DIR",
        BASELINE_RUNS,
        [name for name, _ in NDCG5_RUNS],
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


output_path = Path("notebooks") / "sasrec_timeaware_bpi2012_colab_train_06_260514.ipynb"
output_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(output_path)
