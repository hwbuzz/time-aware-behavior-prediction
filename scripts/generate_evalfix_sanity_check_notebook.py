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


def train_cell(run_name: str, output_dir: str, selection_metric: str, params: dict) -> str:
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


ANCHOR_PARAMS = dict(hidden_units=32, num_blocks=2, num_heads=1, maxlen=20, lr=0.001, dropout_rate=0.2)
REFINE_PARAMS = dict(hidden_units=50, num_blocks=2, num_heads=1, maxlen=50, lr=0.001, dropout_rate=0.35)


ndcg10_existing = [
    "anchor_ml20_s42",
    "refine_ml50_do035_s42",
]
ndcg10_new = [
    ("anchor_ml20_s2024", dict(**ANCHOR_PARAMS, seed=2024)),
    ("anchor_ml20_s7", dict(**ANCHOR_PARAMS, seed=7)),
    ("refine_ml50_do035_s2024", dict(**REFINE_PARAMS, seed=2024)),
    ("refine_ml50_do035_s7", dict(**REFINE_PARAMS, seed=7)),
]

ndcg5_existing = [
    "anchor_ml20_s42",
    "refine_ml50_do035_s42",
]
ndcg5_new = [
    ("anchor_ml20_s2024", dict(**ANCHOR_PARAMS, seed=2024)),
    ("anchor_ml20_s7", dict(**ANCHOR_PARAMS, seed=7)),
    ("refine_ml50_do035_s2024", dict(**REFINE_PARAMS, seed=2024)),
    ("refine_ml50_do035_s7", dict(**REFINE_PARAMS, seed=7)),
]


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


cells = [
    md_cell(
        "# SASRec BPI2012 Colab Sanity Check (Eval Fix)\n\n"
        "Colab notebook for sanity checking the two selected post-fix baseline candidates.\n\n"
        "Goals:\n"
        "- reuse already completed runs instead of retraining them\n"
        "- train only the missing seeds for the two selected candidate settings\n"
        "- summarize mean/std and valid-test trends separately for `NDCG@10` and `NDCG@5` model-selection criteria\n"
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
        "## Candidate settings\n\n"
        "Selected candidates for sanity check:\n"
        "- `anchor_ml20` (`hidden_units=32, maxlen=20, dropout=0.2`)\n"
        "- `refine_ml50_do035` (`hidden_units=50, maxlen=50, dropout=0.35`)\n\n"
        "We will evaluate them under two model-selection criteria separately:\n"
        "- `full_valid_ndcg@10`\n"
        "- `full_valid_ndcg@5`\n"
    ),
    md_cell(
        "## Check existing completed runs\n\n"
        "These runs should already exist and **must not be retrained**.\n"
    ),
    code_cell(
        "from pathlib import Path\n\n"
        "existing_ndcg10 = [\n" +
        "".join([f"    '{name}',\n" for name in ndcg10_existing]) +
        "]\n"
        "existing_ndcg5 = [\n" +
        "".join([f"    '{name}',\n" for name in ndcg5_existing]) +
        "]\n\n"
        "for label, output_dir, run_names in [\n"
        "    ('NDCG@10', Path(NDCG10_OUTPUT_DIR), existing_ndcg10),\n"
        "    ('NDCG@5', Path(NDCG5_OUTPUT_DIR), existing_ndcg5),\n"
        "]:\n"
        "    print('=' * 80)\n"
        "    print(label)\n"
        "    for run_name in run_names:\n"
        "        run_dir = output_dir / run_name\n"
        "        print(run_name, 'EXISTS' if run_dir.exists() else 'MISSING')\n"
    ),
    md_cell(
        "## Train only missing seeds for `NDCG@10`\n\n"
        "Run these cells only if the corresponding run directory does not already exist.\n"
    ),
]

for run_name, params in ndcg10_new:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(train_cell(run_name, NDCG10_OUTPUT, "full_valid_ndcg@10", params)))

cells.append(
    md_cell(
        "## Train only missing seeds for `NDCG@5`\n\n"
        "Run these cells only if the corresponding run directory does not already exist.\n"
    )
)

for run_name, params in ndcg5_new:
    cells.append(md_cell(f"### {run_name}\n"))
    cells.append(code_cell(train_cell(run_name, NDCG5_OUTPUT, "full_valid_ndcg@5", params)))

cells.extend(
    [
        md_cell(
            "## Rebuild result table from run folders\n\n"
            "This avoids schema issues in `experiment_index.csv` and lets us combine existing and newly added runs safely.\n"
        ),
        code_cell(rebuild_df_code()),
        md_cell("## `NDCG@10` sanity check summary\n"),
        code_cell(
            "ndcg10_targets = [\n"
            "    'anchor_ml20_s42',\n"
            "    'anchor_ml20_s2024',\n"
            "    'anchor_ml20_s7',\n"
            "    'refine_ml50_do035_s42',\n"
            "    'refine_ml50_do035_s2024',\n"
            "    'refine_ml50_do035_s7',\n"
            "]\n\n"
            "df10 = rebuild_df(NDCG10_OUTPUT_DIR)\n"
            "df10_sc = df10[df10['run_name'].isin(ndcg10_targets)].copy()\n"
            "df10_sc = df10_sc.sort_values(['run_name']).reset_index(drop=True)\n"
            "df10_sc[[\n"
            "    'run_name', 'seed',\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]]\n"
        ),
        code_cell(
            "df10_sc['candidate'] = df10_sc['run_name'].apply(\n"
            "    lambda x: 'anchor_ml20' if 'anchor_ml20' in x else 'refine_ml50_do035'\n"
            ")\n"
            "summary10 = df10_sc.groupby('candidate')[[\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]].agg(['mean', 'std'])\n"
            "summary10\n"
        ),
        md_cell(
            "Interpretation guide for `NDCG@10`:\n"
            "- compare mean/std of `best_valid_full_ndcg@10` and `best_test_full_ndcg@10`\n"
            "- then check whether `@5`, sampled, and MRR show a similar trend\n"
        ),
        md_cell("## `NDCG@5` sanity check summary\n"),
        code_cell(
            "ndcg5_targets = [\n"
            "    'anchor_ml20_s42',\n"
            "    'anchor_ml20_s2024',\n"
            "    'anchor_ml20_s7',\n"
            "    'refine_ml50_do035_s42',\n"
            "    'refine_ml50_do035_s2024',\n"
            "    'refine_ml50_do035_s7',\n"
            "]\n\n"
            "df5 = rebuild_df(NDCG5_OUTPUT_DIR)\n"
            "df5_sc = df5[df5['run_name'].isin(ndcg5_targets)].copy()\n"
            "df5_sc = df5_sc.sort_values(['run_name']).reset_index(drop=True)\n"
            "df5_sc[[\n"
            "    'run_name', 'seed',\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]]\n"
        ),
        code_cell(
            "df5_sc['candidate'] = df5_sc['run_name'].apply(\n"
            "    lambda x: 'anchor_ml20' if 'anchor_ml20' in x else 'refine_ml50_do035'\n"
            ")\n"
            "summary5 = df5_sc.groupby('candidate')[[\n"
            "    'best_valid_full_ndcg@5', 'best_test_full_ndcg@5',\n"
            "    'best_valid_full_ndcg@10', 'best_test_full_ndcg@10',\n"
            "    'best_valid_full_mrr', 'best_test_full_mrr',\n"
            "    'best_valid_sampled_ndcg@5', 'best_test_sampled_ndcg@5',\n"
            "    'best_valid_sampled_ndcg@10', 'best_test_sampled_ndcg@10',\n"
            "    'best_valid_sampled_mrr', 'best_test_sampled_mrr',\n"
            "]].agg(['mean', 'std'])\n"
            "summary5\n"
        ),
        md_cell(
            "Interpretation guide for `NDCG@5`:\n"
            "- compare mean/std of `best_valid_full_ndcg@5` and `best_test_full_ndcg@5`\n"
            "- then check whether `@10`, sampled, and MRR show a similar trend\n"
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

output_path = Path("notebooks") / "sasrec_bpi2012_colab_train_sanity_check_evalfix_260513.ipynb"
output_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(output_path)
