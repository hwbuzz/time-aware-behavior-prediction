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


COMMON_FLAGS = """--interactions_path data/processed/bpi2012_complete_only/sasrec_interactions.txt \\
  --output_dir "/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg5" \\
  --batch_size 128 \\
  --num_epochs 50 \\
  --eval_every 5 \\
  --device cuda \\
  --num_negative_samples 100 \\
  --eval_protocol both \\
  --topk_list 5,10 \\
  --selection_metric full_valid_ndcg@5 \\
  --save_every_eval"""


RUNS = [
    ("anchor_v3_bpi_regularized_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.3, seed=42)),
    ("anchor_v3_paper_default_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=200, lr=0.001, dropout_rate=0.2, seed=42)),
    ("anchor_v3_bpi_long_context_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=100, lr=0.001, dropout_rate=0.2, seed=42)),
    ("anchor_v3_bpi_mid_context_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.2, seed=42)),
    ("anchor_v3_bpi_short_context_seed42_ndcg5", dict(hidden_units=32, num_blocks=2, num_heads=1, maxlen=20, lr=0.001, dropout_rate=0.2, seed=42)),
    ("refine_v3_ml50_do030_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.3, seed=42)),
    ("refine_v3_ml50_do035_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.35, seed=42)),
    ("refine_v3_ml50_do025_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.25, seed=42)),
    ("refine_v3_ml50_do025_seed2024_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.25, seed=2024)),
    ("refine_v3_ml50_do030_seed2024_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.3, seed=2024)),
    ("refine_v3_ml75_do030_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=75, lr=0.001, dropout_rate=0.3, seed=42)),
    ("refine_v3_ml100_do030_seed42_ndcg5", dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=100, lr=0.001, dropout_rate=0.3, seed=42)),
]


def run_cell(run_name: str, params: dict) -> str:
    flags = "\n".join(
        [
            "!python src/train_sasrec.py \\",
            f'  --run_name {run_name} \\',
            f"  --hidden_units {params['hidden_units']} \\",
            f"  --num_blocks {params['num_blocks']} \\",
            f"  --num_heads {params['num_heads']} \\",
            f"  --maxlen {params['maxlen']} \\",
            f"  --lr {params['lr']} \\",
            f"  --dropout_rate {params['dropout_rate']} \\",
            f"  --seed {params['seed']} \\",
            f"  {COMMON_FLAGS}",
            "",
        ]
    )
    return flags


cells = [
    md_cell(
        "# SASRec BPI2012 Colab Train (`NDCG@5` selection)\n\n"
        "Colab notebook for re-running all `anchor_v3` and `refine_v3` settings with "
        "`selection_metric=full_valid_ndcg@5`.\n\n"
        "Current evaluation protocol:\n"
        "- model selection: `full_valid_ndcg@5`\n"
        "- main evaluation: `full` ranking with `@5`, `@10`, `MRR`\n"
        "- supplementary evaluation: `sampled` ranking with `num_negative_samples=100`, `@5`, `@10`, `MRR`\n"
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
        "OUTPUT_DIR = f'{DRIVE_ROOT}/outputs/sasrec_bpi2012_ndcg5'\n"
        "NOTEBOOK_DIR = f'{DRIVE_ROOT}/notebooks'\n\n"
        "print('DRIVE_ROOT:', DRIVE_ROOT)\n"
        "print('DATA_DIR:', DATA_DIR)\n"
        "print('OUTPUT_DIR:', OUTPUT_DIR)\n"
    ),
    code_cell(
        "!mkdir -p \"$NOTEBOOK_DIR\"\n"
        "!mkdir -p \"$DATA_DIR\"\n"
        "!mkdir -p \"$OUTPUT_DIR\"\n"
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
    code_cell(
        "OUTPUT_DIR = '/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg5'\n"
        "INTERACTIONS_PATH = 'data/processed/bpi2012_complete_only/sasrec_interactions.txt'\n"
    ),
    md_cell(
        "## Planned runs\n\n"
        "These runs share the same evaluation settings:\n"
        "- `--eval_protocol both`\n"
        "- `--topk_list 5,10`\n"
        "- `--selection_metric full_valid_ndcg@5`\n"
        "- `--num_negative_samples 100`\n"
        "- `--save_every_eval`\n"
    ),
    code_cell(
        "from pathlib import Path\n\n"
        "output_dir = Path('/content/drive/MyDrive/ai-projects/time-aware-behavior-prediction/outputs/sasrec_bpi2012_ndcg5')\n"
        "planned_run_names = [\n" +
        "".join([f"    '{name}',\n" for name, _ in RUNS]) +
        "]\n\n"
        "for run_name in planned_run_names:\n"
        "    run_dir = output_dir / run_name\n"
        "    print(run_name, 'EXISTS' if run_dir.exists() else 'OK')\n"
    ),
]

for run_name, params in RUNS:
    cells.append(md_cell(f"## {run_name}\n"))
    cells.append(code_cell(run_cell(run_name, params)))

cells.extend(
    [
        md_cell(
            "## Result lookup\n\n"
            "Sort by `best_valid_full_ndcg@5` first because this notebook uses `NDCG@5` as the model-selection metric.\n"
        ),
        code_cell(
            "import pandas as pd\n\n"
            "index_path = f\"{OUTPUT_DIR}/experiment_index.csv\"\n"
            "df = pd.read_csv(index_path)\n\n"
            "df = df.sort_values(\n"
            "    ['best_valid_full_ndcg@5', 'best_valid_full_hr@5'],\n"
            "    ascending=False\n"
            ").reset_index(drop=True)\n\n"
            "df[[\n"
            "    'run_name',\n"
            "    'best_epoch',\n"
            "    'best_valid_full_ndcg@5',\n"
            "    'best_valid_full_hr@5',\n"
            "    'best_valid_full_ndcg@10',\n"
            "    'best_valid_full_hr@10',\n"
            "    'best_valid_full_mrr',\n"
            "    'best_test_full_ndcg@5',\n"
            "    'best_test_full_hr@5',\n"
            "    'best_test_full_ndcg@10',\n"
            "    'best_test_full_hr@10',\n"
            "    'best_test_full_mrr',\n"
            "    'checkpoint_best',\n"
            "    'checkpoint_last',\n"
            "]]\n"
        ),
        code_cell(
            "df[[\n"
            "    'run_name',\n"
            "    'best_valid_sampled_ndcg@5',\n"
            "    'best_valid_sampled_hr@5',\n"
            "    'best_valid_sampled_ndcg@10',\n"
            "    'best_valid_sampled_hr@10',\n"
            "    'best_valid_sampled_mrr',\n"
            "    'best_test_sampled_ndcg@5',\n"
            "    'best_test_sampled_hr@5',\n"
            "    'best_test_sampled_ndcg@10',\n"
            "    'best_test_sampled_hr@10',\n"
            "    'best_test_sampled_mrr',\n"
            "]].head(20)\n"
        ),
        code_cell(
            "best_row = df.iloc[0]\n"
            "best_row\n"
        ),
        code_cell(
            "print('best run_name:', best_row['run_name'])\n"
            "print('best checkpoint:', best_row['checkpoint_best'])\n"
            "print('last checkpoint:', best_row['checkpoint_last'])\n"
            "print('selection metric:', best_row['best_valid_full_ndcg@5'])\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n\n"
            "config = json.loads(Path(best_row['config_path']).read_text(encoding='utf-8'))\n"
            "summary = json.loads(Path(best_row['metrics_summary']).read_text(encoding='utf-8'))\n\n"
            "print('config')\n"
            "display(config)\n"
            "print('summary')\n"
            "display(summary)\n"
        ),
        code_cell(
            "history = pd.read_csv(best_row['metrics_history'])\n"
            "display(history)\n"
        ),
        code_cell(
            "ax = history.plot(\n"
            "    x='epoch',\n"
            "    y=['full_valid_ndcg@5', 'full_valid_ndcg@10', 'full_test_ndcg@5', 'full_test_ndcg@10'],\n"
            "    marker='o',\n"
            "    figsize=(10, 4),\n"
            "    title=f\"Full-ranking NDCG by epoch: {best_row['run_name']}\"\n"
            ")\n"
            "ax.grid(True, alpha=0.3)\n"
        ),
    ]
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

output_path = Path("notebooks") / "sasrec_bpi2012_colab_train_ndcg5_260509.ipynb"
output_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(output_path)
