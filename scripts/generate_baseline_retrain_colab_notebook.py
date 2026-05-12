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


NDCG10_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg10"
NDCG5_OUTPUT = "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg5"


RUNS = [
    ("anchor_pd_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=200, lr=0.001, dropout_rate=0.2, seed=42)),
    ("anchor_ml100_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=100, lr=0.001, dropout_rate=0.2, seed=42)),
    ("anchor_ml50_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.2, seed=42)),
    ("anchor_ml20_s42", dict(hidden_units=32, num_blocks=2, num_heads=1, maxlen=20, lr=0.001, dropout_rate=0.2, seed=42)),
    ("refine_ml50_do030_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.3, seed=42)),
    ("refine_ml50_do035_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.35, seed=42)),
    ("refine_ml50_do025_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.25, seed=42)),
    ("refine_ml50_do025_s2024", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.25, seed=2024)),
    ("refine_ml50_do030_s2024", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.3, seed=2024)),
    ("refine_ml75_do030_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=75, lr=0.001, dropout_rate=0.3, seed=42)),
    ("refine_ml100_do030_s42", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=100, lr=0.001, dropout_rate=0.3, seed=42)),
]


def run_cell(run_name: str, output_dir: str, selection_metric: str, params: dict) -> str:
    return "\n".join(
        [
            "!python src/train_sasrec.py \\",
            f"  --run_name {run_name} \\",
            f"  --hidden_units {params['hidden_units']} \\",
            f"  --num_blocks {params['num_blocks']} \\",
            f"  --num_heads {params['num_heads']} \\",
            f"  --maxlen {params['maxlen']} \\",
            f"  --lr {params['lr']} \\",
            f"  --dropout_rate {params['dropout_rate']} \\",
            f"  --seed {params['seed']} \\",
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
        "        }\n"
        "        best_valid = summary.get('best_valid', {})\n"
        "        best_test = summary.get('best_test_at_best_valid', {})\n"
        "        def pick(metrics_group, mode, key):\n"
        "            return metrics_group.get(mode, {}).get(key)\n"
        "        row.update({\n"
        "            'best_valid_full_ndcg@5': pick(best_valid, 'full', 'ndcg@5'),\n"
        "            'best_valid_full_hr@5': pick(best_valid, 'full', 'hr@5'),\n"
        "            'best_valid_full_ndcg@10': pick(best_valid, 'full', 'ndcg@10'),\n"
        "            'best_valid_full_hr@10': pick(best_valid, 'full', 'hr@10'),\n"
        "            'best_valid_full_mrr': pick(best_valid, 'full', 'mrr'),\n"
        "            'best_test_full_ndcg@5': pick(best_test, 'full', 'ndcg@5'),\n"
        "            'best_test_full_hr@5': pick(best_test, 'full', 'hr@5'),\n"
        "            'best_test_full_ndcg@10': pick(best_test, 'full', 'ndcg@10'),\n"
        "            'best_test_full_hr@10': pick(best_test, 'full', 'hr@10'),\n"
        "            'best_test_full_mrr': pick(best_test, 'full', 'mrr'),\n"
        "            'best_valid_sampled_ndcg@5': pick(best_valid, 'sampled', 'ndcg@5'),\n"
        "            'best_valid_sampled_hr@5': pick(best_valid, 'sampled', 'hr@5'),\n"
        "            'best_valid_sampled_ndcg@10': pick(best_valid, 'sampled', 'ndcg@10'),\n"
        "            'best_valid_sampled_hr@10': pick(best_valid, 'sampled', 'hr@10'),\n"
        "            'best_valid_sampled_mrr': pick(best_valid, 'sampled', 'mrr'),\n"
        "            'best_test_sampled_ndcg@5': pick(best_test, 'sampled', 'ndcg@5'),\n"
        "            'best_test_sampled_hr@5': pick(best_test, 'sampled', 'hr@5'),\n"
        "            'best_test_sampled_ndcg@10': pick(best_test, 'sampled', 'ndcg@10'),\n"
        "            'best_test_sampled_hr@10': pick(best_test, 'sampled', 'hr@10'),\n"
        "            'best_test_sampled_mrr': pick(best_test, 'sampled', 'mrr'),\n"
        "        })\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows)\n"
    )


def result_section(label: str, output_dir_var: str, selection_col: str) -> list[dict]:
    sort_cols = (
        "['best_valid_full_ndcg@10', 'best_valid_full_hr@10']"
        if selection_col == "best_valid_full_ndcg@10"
        else "['best_valid_full_ndcg@5', 'best_valid_full_hr@5']"
    )
    return [
        md_cell(f"## Result lookup: {label}\n"),
        code_cell(
            f"df_{label.lower()} = rebuild_df({output_dir_var})\n"
            f"df_{label.lower()} = df_{label.lower()}.sort_values({sort_cols}, ascending=False).reset_index(drop=True)\n"
            f"df_{label.lower()}[[\n"
            "    'run_name', 'seed', 'maxlen', 'dropout_rate', 'best_epoch',\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "    'checkpoint_best', 'checkpoint_last',\n"
            "]]\n"
        ),
    ]


cells = [
    md_cell(
        "# SASRec BPI2012 Colab Train (Eval Fix)\n\n"
        "Colab notebook for re-running the selected SASRec baseline settings after fixing the evaluation code.\n\n"
        "This notebook runs both model-selection criteria:\n"
        "- `full_valid_ndcg@10`\n"
        "- `full_valid_ndcg@5`\n\n"
        "Output directories:\n"
        "- `sasrec_bpi2012_ndcg10`\n"
        "- `sasrec_bpi2012_ndcg5`\n"
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
        f"NDCG10_OUTPUT_DIR = '{NDCG10_OUTPUT}'\n"
        f"NDCG5_OUTPUT_DIR = '{NDCG5_OUTPUT}'\n"
        "NOTEBOOK_DIR = f'{DRIVE_ROOT}/notebooks'\n\n"
        "print('DATA_DIR:', DATA_DIR)\n"
        "print('NDCG10_OUTPUT_DIR:', NDCG10_OUTPUT_DIR)\n"
        "print('NDCG5_OUTPUT_DIR:', NDCG5_OUTPUT_DIR)\n"
    ),
    code_cell(
        "!mkdir -p \"$NOTEBOOK_DIR\"\n"
        "!mkdir -p \"$DATA_DIR\"\n"
        "!mkdir -p \"$NDCG10_OUTPUT_DIR\"\n"
        "!mkdir -p \"$NDCG5_OUTPUT_DIR\"\n"
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
        "## Planned runs\n\n"
        "Selected baseline runs to re-run after the evaluation fix:\n"
        "- `anchor_pd_s42`\n"
        "- `anchor_ml100_s42`\n"
        "- `anchor_ml50_s42`\n"
        "- `anchor_ml20_s42`\n"
        "- `refine_ml50_do030_s42`\n"
        "- `refine_ml50_do035_s42`\n"
        "- `refine_ml50_do025_s42`\n"
        "- `refine_ml50_do025_s2024`\n"
        "- `refine_ml50_do030_s2024`\n"
        "- `refine_ml75_do030_s42`\n"
        "- `refine_ml100_do030_s42`\n"
    ),
    code_cell(
        "from pathlib import Path\n\n"
        "planned_run_names = [\n" +
        "".join([f"    '{name}',\n" for name, _ in RUNS]) +
        "]\n\n"
        "for label, output_dir in [('NDCG@10', Path(NDCG10_OUTPUT_DIR)), ('NDCG@5', Path(NDCG5_OUTPUT_DIR))]:\n"
        "    print('=' * 80)\n"
        "    print(label)\n"
        "    for run_name in planned_run_names:\n"
        "        run_dir = output_dir / run_name\n"
        "        print(run_name, 'EXISTS' if run_dir.exists() else 'OK')\n"
    ),
    md_cell(
        "## Train runs with `selection_metric = full_valid_ndcg@10`\n"
    ),
]

for run_name, params in RUNS:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(run_cell(run_name, NDCG10_OUTPUT, "full_valid_ndcg@10", params)))

cells.append(md_cell("## Train runs with `selection_metric = full_valid_ndcg@5`\n"))

for run_name, params in RUNS:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(run_cell(run_name, NDCG5_OUTPUT, "full_valid_ndcg@5", params)))

cells.extend(
    [
        md_cell(
            "## Result lookup\n\n"
            "Rebuild result tables directly from each run folder to avoid any CSV schema drift.\n"
        ),
        code_cell(rebuild_df_code()),
    ]
)

cells.extend(result_section("ndcg10", "NDCG10_OUTPUT_DIR", "best_valid_full_ndcg@10"))
cells.extend(result_section("ndcg5", "NDCG5_OUTPUT_DIR", "best_valid_full_ndcg@5"))


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

output_path = Path("notebooks") / "sasrec_bpi2012_colab_train_evalfix_260512.ipynb"
output_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(output_path)
