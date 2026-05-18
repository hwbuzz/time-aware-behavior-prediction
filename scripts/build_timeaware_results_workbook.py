from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
DOCS = ROOT / "docs"

NOTEBOOK_FILES = [
    "sasrec_timeaware_bpi2012_colab_train_02_260513.ipynb",
    "sasrec_timeaware_bpi2012_colab_train_03_260513.ipynb",
    "sasrec_timeaware_bpi2012_colab_train_04_260514.ipynb",
    "sasrec_timeaware_bpi2012_colab_train_05_260514.ipynb",
    "sasrec_timeaware_bpi2012_colab_train_06_260514.ipynb",
    "sasrec_timeaware_bpi2012_colab_train_07_260514.ipynb",
    "sasrec_timeaware_bpi2012_colab_train_08_260515.ipynb",
]

NUMERIC_COLS = [
    "best_valid_full_ndcg@10",
    "best_valid_full_hr@10",
    "best_valid_full_ndcg@5",
    "best_valid_full_hr@5",
    "best_valid_full_mrr",
    "best_test_full_ndcg@10",
    "best_test_full_hr@10",
    "best_test_full_ndcg@5",
    "best_test_full_hr@5",
    "best_test_full_mrr",
    "best_valid_sampled_ndcg@10",
    "best_valid_sampled_hr@10",
    "best_valid_sampled_ndcg@5",
    "best_valid_sampled_hr@5",
    "best_valid_sampled_mrr",
    "best_test_sampled_ndcg@10",
    "best_test_sampled_hr@10",
    "best_test_sampled_ndcg@5",
    "best_test_sampled_hr@5",
    "best_test_sampled_mrr",
]


def parse_detail_table_from_output(output: dict) -> pd.DataFrame:
    text = "".join(output.get("data", {}).get("text/plain", []))
    df = pd.read_fwf(io.StringIO(text))

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "bucket_variant" in df.columns and "time_variant" not in df.columns:
        df = df.rename(columns={"bucket_variant": "time_variant"})
    if "time_bucket_boundaries" in df.columns and "time_bucket_boundaries_parsed" not in df.columns:
        df = df.rename(columns={"time_bucket_boundaries": "time_bucket_boundaries_parsed"})

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if "time_bucket_boundaries_parsed" in df.columns and unnamed_cols:
        for col in unnamed_cols:
            if df[col].notna().any():
                df["time_bucket_boundaries_parsed"] = (
                    df[col].fillna("").astype(str).str.strip()
                    + " "
                    + df["time_bucket_boundaries_parsed"].fillna("").astype(str).str.strip()
                ).str.strip()
        df = df.drop(columns=unnamed_cols)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_detail_frames(notebook_path: Path) -> list[pd.DataFrame]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        selection_basis = None
        if "df_ndcg10[[" in source:
            selection_basis = "ndcg10"
        elif "df_ndcg5[[" in source:
            selection_basis = "ndcg5"

        if selection_basis is None:
            continue

        outputs = cell.get("outputs", [])
        table_output = next((o for o in outputs if "data" in o and "text/plain" in o.get("data", {})), None)
        if table_output is None:
            continue

        df = parse_detail_table_from_output(table_output)
        df["notebook"] = notebook_path.name
        df["selection_basis"] = selection_basis
        df["experiment_name"] = notebook_path.stem
        frames.append(df)

    return frames


def build_detail_df() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in NOTEBOOK_FILES:
        path = NOTEBOOKS / name
        if not path.exists():
            continue
        frames.extend(extract_detail_frames(path))

    if not frames:
        return pd.DataFrame()

    detail_df = pd.concat(frames, ignore_index=True)
    return detail_df


def build_summary_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    summary = (
        detail_df.groupby(
            ["notebook", "experiment_name", "selection_basis", "time_variant"],
            dropna=False,
        )[NUMERIC_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        f"{metric}_{agg}" if agg else metric
        for metric, agg in [
            (col if isinstance(col, str) else col[0], "" if isinstance(col, str) else col[1])
            for col in summary.columns
        ]
    ]
    return summary


def main() -> None:
    detail_df = build_detail_df()
    summary_df = build_summary_df(detail_df)

    output_path = DOCS / "timeaware_results_02_08_260517.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary_mean_std")
        detail_df.to_excel(writer, index=False, sheet_name="detail_runs")

    print(output_path)
    print(f"summary_rows={len(summary_df)} detail_rows={len(detail_df)}")


if __name__ == "__main__":
    main()
