from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from shutil import copy2
from typing import Iterable

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill
from openpyxl.utils import get_column_letter


WORKBOOK_PATH = Path("docs/캡스톤_모델성능_v0.5_260604.xlsx")

STAGE3_NOTEBOOKS = {
    "02": Path("notebooks/sasrec_stage3_bpi2012_colab_train_02_260601.ipynb"),
    "03": Path("notebooks/sasrec_stage3_bpi2012_colab_train_03_260601.ipynb"),
    "04": Path("notebooks/sasrec_stage3_bpi2012_colab_train_04_260601.ipynb"),
    "05": Path("notebooks/sasrec_stage3_bpi2012_colab_train_05_260602.ipynb"),
    "06": Path("notebooks/sasrec_stage3_bpi2012_colab_train_06_260602.ipynb"),
    "07": Path("notebooks/sasrec_stage3_bpi2012_time_naive_baselines_07_260602.ipynb"),
}

NEW_SHEETS = [
    "3-0.Stage3",
    "3-1.Baseline_mt",
    "3-2.AttnBias_mt",
    "3-3.AttnBias_w01",
    "3-4.Anchor_w01",
    "3-5.Refine_w01",
    "3-6.Time_naive",
]

SUMMARY_METRICS = [
    "best_valid_full_ndcg@10",
    "best_valid_full_hr@10",
    "best_valid_full_ndcg@5",
    "best_valid_full_hr@5",
    "best_valid_full_mrr",
    "best_test_at_best_valid_full_ndcg@10",
    "best_test_at_best_valid_full_hr@10",
    "best_test_at_best_valid_full_ndcg@5",
    "best_test_at_best_valid_full_hr@5",
    "best_test_at_best_valid_full_mrr",
    "best_valid_sampled_ndcg@10",
    "best_valid_sampled_hr@10",
    "best_valid_sampled_ndcg@5",
    "best_valid_sampled_hr@5",
    "best_valid_sampled_mrr",
    "best_test_at_best_valid_sampled_ndcg@10",
    "best_test_at_best_valid_sampled_hr@10",
    "best_test_at_best_valid_sampled_ndcg@5",
    "best_test_at_best_valid_sampled_hr@5",
    "best_test_at_best_valid_sampled_mrr",
    "best_valid_task_accuracy",
    "best_valid_task_macro_f1",
    "best_valid_task_top5_accuracy",
    "best_valid_task_top10_accuracy",
    "best_valid_task_time_mae",
    "best_valid_task_time_rmse",
    "best_valid_task_time_median_ae",
    "best_test_at_best_valid_task_accuracy",
    "best_test_at_best_valid_task_macro_f1",
    "best_test_at_best_valid_task_top5_accuracy",
    "best_test_at_best_valid_task_top10_accuracy",
    "best_test_at_best_valid_task_time_mae",
    "best_test_at_best_valid_task_time_rmse",
    "best_test_at_best_valid_task_time_median_ae",
]


class SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._in_cell = False
        self._text = ""
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"}:
            self._in_cell = True
            self._text = ""
            self._colspan = int(attr_map.get("colspan", "1") or "1")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._in_cell:
            text = " ".join(self._text.split())
            if self._row is not None:
                self._row.extend([text] * self._colspan)
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._text += data


@dataclass
class NotebookTables:
    detail: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    naive_readable: pd.DataFrame | None = None
    model_summary: pd.DataFrame | None = None


def parse_html_rows(html_parts: str | list[str]) -> list[list[str]]:
    html = "".join(html_parts) if isinstance(html_parts, list) else html_parts
    parser = SimpleHTMLTableParser()
    parser.feed(html)
    return parser.rows


def rows_to_df(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    if len(rows) >= 3 and len(rows[1]) > 1 and rows[1][1] in {"mean", "std"}:
        index_name = rows[2][0] or rows[0][0] or "index"
        columns = [index_name]
        for metric, stat in zip(rows[0][1:], rows[1][1:]):
            columns.append(f"{metric}|{stat}")
        data = [r[: len(columns)] for r in rows[3:]]
        return pd.DataFrame(data, columns=columns)

    header = rows[0]
    data = [r[: len(header)] for r in rows[1:]]
    return pd.DataFrame(data, columns=header)


def load_notebook_tables(nb_path: Path) -> NotebookTables:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    result = NotebookTables()

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])
        html_outputs = []
        for output in outputs:
            data = output.get("data", {})
            if "text/html" in data:
                html_outputs.append(data["text/html"])
        if not html_outputs:
            continue

        if "df_compare[existing_display_cols]" in source or "df_compare[display_cols]" in source:
            result.detail = rows_to_df(parse_html_rows(html_outputs[0]))
        elif "summary_compare =" in source:
            result.summary = rows_to_df(parse_html_rows(html_outputs[0]))
        elif "readable" in source and "baseline_results" in source:
            result.naive_readable = rows_to_df(parse_html_rows(html_outputs[0]))
        elif "def rebuild_df" in source:
            result.model_summary = rows_to_df(parse_html_rows(html_outputs[0]))

    return result


def maybe_numeric(value: str):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if text in {"None", "nan", "NaN"}:
        return None
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def clean_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "" in out.columns:
        out = out.drop(columns=[""])
    for col in out.columns:
        out[col] = out[col].map(maybe_numeric)
    return out


def clean_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    index_col = out.columns[0]
    out = out.rename(columns={index_col: "variant"})
    for col in out.columns[1:]:
        out[col] = out[col].map(maybe_numeric)
    filtered_cols = ["variant"]
    for metric in SUMMARY_METRICS:
        for stat in ("mean", "std"):
            key = f"{metric}|{stat}"
            if key in out.columns:
                filtered_cols.append(key)
    return out[filtered_cols]


def clean_naive_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "" in out.columns:
        out = out.drop(columns=[""])
    for col in out.columns:
        out[col] = out[col].map(maybe_numeric)
    return out


def get_metric(summary_df: pd.DataFrame, variant: str, metric: str, stat: str = "mean"):
    if summary_df.empty:
        return None
    key = f"{metric}|{stat}"
    rows = summary_df[summary_df["variant"] == variant]
    if rows.empty or key not in rows.columns:
        return None
    return rows.iloc[0][key]


def fnum(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def fsec_to_hours(value) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value) / 3600:.2f}h"


def clone_cell_style(src, dst) -> None:
    dst.font = copy(src.font)
    dst.fill = copy(src.fill)
    dst.border = copy(src.border)
    dst.alignment = copy(src.alignment)
    dst.number_format = src.number_format
    dst.protection = copy(src.protection)


def apply_template_styles(ws, template_ws) -> None:
    clone_cell_style(template_ws["B2"], ws["B2"])
    clone_cell_style(template_ws["B4"], ws["B4"])
    clone_cell_style(template_ws["C6"], ws["C6"])
    clone_cell_style(template_ws["D6"], ws["D6"])
    clone_cell_style(template_ws["B12"], ws["B12"])
    clone_cell_style(template_ws["B13"], ws["B13"])
    for col in range(2, 60):
        clone_cell_style(template_ws.cell(16, min(col, template_ws.max_column)), ws.cell(16, col))
        clone_cell_style(template_ws.cell(57, min(col, template_ws.max_column)), ws.cell(57, col))
    clone_cell_style(template_ws["I55"], ws["B55"])
    clone_cell_style(template_ws["S55"], ws["T55"])


def set_default_layout(ws) -> None:
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B17"
    ws.column_dimensions["A"].width = 3
    for col in range(2, 60):
        ws.column_dimensions[get_column_letter(col)].width = 14
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 34
    ws.column_dimensions["E"].width = 16
    ws.column_dimensions["F"].width = 14
    ws.column_dimensions["G"].width = 18
    ws.column_dimensions["H"].width = 18


def write_summary_block(ws, title: str, rows: list[tuple[str, str]], bullets: list[str], max_col: int = 28) -> int:
    end_col_letter = get_column_letter(max_col)
    ws["B2"] = "SUMMARY"
    ws.merge_cells(f"B4:{end_col_letter}4")
    ws["B4"] = title

    current_row = 6
    for key, value in rows:
        ws[f"C{current_row}"] = key
        ws[f"D{current_row}"] = value
        current_row += 1

    current_row += 1
    for bullet in bullets:
        ws.merge_cells(f"B{current_row}:{end_col_letter}{current_row}")
        ws[f"B{current_row}"] = f"▶ {bullet}"
        current_row += 1
    return current_row + 1


def write_dataframe(ws, start_row: int, start_col: int, df: pd.DataFrame, title: str | None = None) -> int:
    row = start_row
    if title:
        ws.cell(row, start_col, title)
        ws.cell(row, start_col).font = Font(bold=True)
        row += 1

    for idx, col in enumerate(df.columns, start_col):
        cell = ws.cell(row, idx, col)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9E1F2")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(
            left=copy(ws["B16"].border.left),
            right=copy(ws["B16"].border.right),
            top=copy(ws["B16"].border.top),
            bottom=copy(ws["B16"].border.bottom),
        )
    row += 1

    for _, record in df.iterrows():
        for idx, col in enumerate(df.columns, start_col):
            val = record[col]
            cell = ws.cell(row, idx, val)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        row += 1
    return row


def build_overview_df(parsed: dict[str, NotebookTables]) -> pd.DataFrame:
    rows = []
    s02 = clean_summary_df(parsed["02"].summary)
    s03 = clean_summary_df(parsed["03"].summary)
    s04 = clean_summary_df(parsed["04"].summary)
    s05 = clean_summary_df(parsed["05"].summary)
    s06 = clean_summary_df(parsed["06"].summary)

    key_variants = [
        ("02", "anchor_single_task", "single-task baseline"),
        ("02", "anchor_multi_task", "plain multitask"),
        ("03", "anchor_attnbias_multi_task", "attnbias multitask"),
        ("05", "anchor_multi_task_w0.1", "plain multitask w0.1"),
        ("06", "refine_multi_task_w0.1", "refine multitask w0.1"),
    ]
    for exp_no, variant, note in key_variants:
        sdf = {"02": s02, "03": s03, "04": s04, "05": s05, "06": s06}[exp_no]
        rows.append(
            {
                "실험": exp_no,
                "variant": variant,
                "설명": note,
                "test_full_ndcg@10_mean": get_metric(sdf, variant, "best_test_at_best_valid_full_ndcg@10", "mean"),
                "test_full_ndcg@10_std": get_metric(sdf, variant, "best_test_at_best_valid_full_ndcg@10", "std"),
                "test_time_mae_mean": get_metric(sdf, variant, "best_test_at_best_valid_task_time_mae", "mean"),
                "test_time_mae_std": get_metric(sdf, variant, "best_test_at_best_valid_task_time_mae", "std"),
            }
        )
    return pd.DataFrame(rows)


def create_stage3_overview(ws, template_ws, parsed: dict[str, NotebookTables]) -> None:
    apply_template_styles(ws, template_ws)
    set_default_layout(ws)

    s02 = clean_summary_df(parsed["02"].summary)
    s03 = clean_summary_df(parsed["03"].summary)
    s05 = clean_summary_df(parsed["05"].summary)
    s06 = clean_summary_df(parsed["06"].summary)

    bullets = [
        (
            "single-task에서 multi-task로 가면 next activity 성능이 전반적으로 낮아졌고, "
            "time_loss_weight를 0.1로 낮추면 activity 성능은 회복되지만 time MAE와 trade-off가 발생함."
        ),
        (
            "Stage 3 best multitask(activity 기준)는 anchor_multi_task_w0.1 "
            f"(test full NDCG@10={fnum(get_metric(s05, 'anchor_multi_task_w0.1', 'best_test_at_best_valid_full_ndcg@10', 'mean'))})"
            "이며, best time MAE는 anchor_attnbias_multi_task "
            f"(MAE={fnum(get_metric(s03, 'anchor_attnbias_multi_task', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초)임."
        ),
        (
            "refine에서도 w0.1 조정 효과가 재현되었지만, overall main metric 기준으로는 "
            f"anchor_multi_task_w0.1 ({fnum(get_metric(s05, 'anchor_multi_task_w0.1', 'best_test_at_best_valid_full_ndcg@10', 'mean'))})"
            f" > refine_multi_task_w0.1 ({fnum(get_metric(s06, 'refine_multi_task_w0.1', 'best_test_at_best_valid_full_ndcg@10', 'mean'))})."
        ),
    ]
    rows = [
        ("실험 단계", "Stage 3: next activity + next time multitask"),
        ("main activity metric", "full ranking, NDCG@10"),
        ("main time metric", "MAE"),
        ("실험 범위", "02~07 (baseline multitask, attention bias multitask, loss-weight, naive baselines)"),
        ("핵심 질문", "multitask가 next activity를 유지하면서 next time도 의미 있게 예측할 수 있는지 확인"),
    ]
    next_row = write_summary_block(ws, "Stage 3 Multitask 실험 개요", rows, bullets, max_col=24)

    overview_df = build_overview_df(parsed)
    next_row = write_dataframe(ws, next_row + 1, 2, overview_df, "핵심 variant 비교")

    naive = clean_naive_df(parsed["07"].naive_readable)
    if not naive.empty:
        test_naive = naive[naive["split"] == "test"].copy()
        test_naive = test_naive[["baseline", "mae", "rmse", "median_ae", "mae_hours", "rmse_hours", "median_ae_minutes"]]
        write_dataframe(ws, next_row + 2, 2, test_naive, "Naive baseline (test split)")


def build_experiment_summary_lines(exp_no: str, summary_df: pd.DataFrame) -> tuple[str, list[tuple[str, str]], list[str]]:
    if exp_no == "02":
        title = "Stage 3 baseline multitask: single-task vs multitask"
        rows = [
            ("모델 구조", "single-task baseline(anchor/refine) vs plain multitask"),
            ("비교 대상", "anchor_single_task, anchor_multi_task_w1.0, refine_single_task, refine_multi_task_w1.0"),
            ("main activity metric", "full ranking, NDCG@10"),
            ("main time metric", "MAE"),
        ]
        bullets = [
            f"anchor 기준으로 single-task({fnum(get_metric(summary_df, 'anchor_single_task', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}) > multitask({fnum(get_metric(summary_df, 'anchor_multi_task', 'best_test_at_best_valid_full_ndcg@10', 'mean'))})",
            f"refine 기준으로도 single-task({fnum(get_metric(summary_df, 'refine_single_task', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}) > multitask({fnum(get_metric(summary_df, 'refine_multi_task', 'best_test_at_best_valid_full_ndcg@10', 'mean'))})",
            "즉 baseline multitask 자체는 next activity 성능을 개선하지 못했고, multitask 내부에서는 anchor backbone이 refine보다 더 안정적이었음.",
        ]
        return title, rows, bullets

    if exp_no == "03":
        title = "Stage 3 attention-bias multitask (anchor backbone)"
        rows = [
            ("모델 구조", "anchor baseline / anchor attention bias / anchor multitask / anchor attnbias multitask"),
            ("비교 대상", "anchor_single_task, anchor_attnbias_single_task, anchor_multi_task_w1.0, anchor_attnbias_multi_task_w1.0"),
            ("time-aware", "delta_start + 9-bucket attention bias"),
            ("main activity metric", "full ranking, NDCG@10"),
        ]
        bullets = [
            f"anchor_attnbias_multi_task의 time MAE({fnum(get_metric(summary_df, 'anchor_attnbias_multi_task', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초)는 plain multitask보다 좋았음.",
            f"하지만 next activity 기준으로는 attnbias multitask({fnum(get_metric(summary_df, 'anchor_attnbias_multi_task', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}) < plain multitask({fnum(get_metric(summary_df, 'anchor_multi_task', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}).",
            "즉 attention bias는 Stage 3에서 time 쪽은 일부 개선했지만, overall activity ranking은 더 악화시킴.",
        ]
        return title, rows, bullets

    if exp_no == "04":
        title = "Stage 3 attention-bias multitask: time_loss_weight 1.0 vs 0.1"
        rows = [
            ("모델 구조", "anchor attention-bias multitask"),
            ("비교 대상", "anchor_attnbias_multi_task_w1.0 vs anchor_attnbias_multi_task_w0.1"),
            ("조정 변수", "time_loss_weight"),
            ("main time metric", "MAE"),
        ]
        bullets = [
            f"w0.1 적용 시 next activity는 일부 회복({fnum(get_metric(summary_df, 'anchor_attnbias_multi_task_w0.1', 'best_test_at_best_valid_full_ndcg@10', 'mean'))} vs {fnum(get_metric(summary_df, 'anchor_attnbias_multi_task_w1.0', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}).",
            f"반대로 time MAE는 악화({fnum(get_metric(summary_df, 'anchor_attnbias_multi_task_w0.1', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초 vs {fnum(get_metric(summary_df, 'anchor_attnbias_multi_task_w1.0', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초).",
            "즉 attention-bias multitask에서도 activity-time trade-off가 뚜렷하게 확인됨.",
        ]
        return title, rows, bullets

    if exp_no == "05":
        title = "Stage 3 plain anchor multitask: time_loss_weight 1.0 vs 0.1"
        rows = [
            ("모델 구조", "anchor baseline single-task vs anchor plain multitask"),
            ("비교 대상", "anchor_single_task, anchor_multi_task_w1.0, anchor_multi_task_w0.1"),
            ("조정 변수", "time_loss_weight"),
            ("main activity metric", "full ranking, NDCG@10"),
        ]
        bullets = [
            f"w0.1 적용으로 anchor multitask activity ranking이 크게 회복됨 ({fnum(get_metric(summary_df, 'anchor_multi_task_w0.1', 'best_test_at_best_valid_full_ndcg@10', 'mean'))} vs {fnum(get_metric(summary_df, 'anchor_multi_task_w1.0', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}).",
            f"하지만 time MAE는 악화됨 ({fnum(get_metric(summary_df, 'anchor_multi_task_w0.1', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초 vs {fnum(get_metric(summary_df, 'anchor_multi_task_w1.0', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초).",
            "즉 plain multitask의 핵심 이슈는 backbone 자체보다도 loss balance에 가까웠고, Stage 3 best multitask(activity 기준)는 anchor_multi_task_w0.1임.",
        ]
        return title, rows, bullets

    if exp_no == "06":
        title = "Stage 3 plain refine multitask: time_loss_weight 1.0 vs 0.1"
        rows = [
            ("모델 구조", "refine baseline single-task vs refine plain multitask"),
            ("비교 대상", "refine_single_task, refine_multi_task_w1.0, refine_multi_task_w0.1"),
            ("조정 변수", "time_loss_weight"),
            ("main activity metric", "full ranking, NDCG@10"),
        ]
        bullets = [
            f"refine에서도 w0.1 적용으로 ranking이 회복됨 ({fnum(get_metric(summary_df, 'refine_multi_task_w0.1', 'best_test_at_best_valid_full_ndcg@10', 'mean'))} vs {fnum(get_metric(summary_df, 'refine_multi_task_w1.0', 'best_test_at_best_valid_full_ndcg@10', 'mean'))}).",
            f"time MAE는 소폭 악화({fnum(get_metric(summary_df, 'refine_multi_task_w0.1', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초 vs {fnum(get_metric(summary_df, 'refine_multi_task_w1.0', 'best_test_at_best_valid_task_time_mae', 'mean'), 1)}초)했지만, anchor처럼 loss-balance 효과가 재현됨.",
            "다만 overall main metric 기준으로는 refine_single_task와 anchor_multi_task_w0.1를 넘지 못함.",
        ]
        return title, rows, bullets

    raise ValueError(f"Unsupported experiment no: {exp_no}")


def select_detail_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "run_name",
        "seed",
        "variant",
        "maxlen",
        "dropout_rate",
        "selection_metric",
        "time_loss_weight",
        "best_valid_full_ndcg@10",
        "best_valid_full_hr@10",
        "best_valid_full_mrr",
        "best_test_at_best_valid_full_ndcg@10",
        "best_test_at_best_valid_full_hr@10",
        "best_test_at_best_valid_full_mrr",
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols].copy()


def create_stage3_experiment_sheet(ws, template_ws, exp_no: str, parsed: NotebookTables) -> None:
    apply_template_styles(ws, template_ws)
    set_default_layout(ws)

    detail = select_detail_columns(clean_detail_df(parsed.detail))
    summary = clean_summary_df(parsed.summary)

    title, summary_rows, bullets = build_experiment_summary_lines(exp_no, summary)
    next_row = write_summary_block(ws, title, summary_rows, bullets, max_col=34)

    if not summary.empty:
        summary = summary.copy()
        metric_order = ["variant"]
        for metric in SUMMARY_METRICS:
            for stat in ("mean", "std"):
                key = f"{metric}|{stat}"
                if key in summary.columns:
                    metric_order.append(key)
        summary = summary[metric_order]
        next_row = write_dataframe(ws, next_row + 1, 2, summary, "전체 지표 mean/std summary")

    if not detail.empty:
        detail = detail.copy()
        detail.insert(0, "SEQ", range(1, len(detail) + 1))
        detail.insert(1, "Dataset", "BPI 2012")
        note_row = next_row + 1
        ws.merge_cells(start_row=note_row, start_column=2, end_row=note_row, end_column=18)
        ws.cell(note_row, 2, "참고: 아래 run별 raw 표는 notebook 저장 시점에 남아 있는 출력 컬럼 기준입니다. 전체 지표는 위 mean/std 표를 기준으로 확인합니다.")
        ws.cell(note_row, 2).font = Font(italic=True)
        next_row = write_dataframe(ws, note_row + 2, 2, detail, "run별 raw 결과")


def create_stage3_naive_sheet(ws, template_ws, parsed07: NotebookTables) -> None:
    apply_template_styles(ws, template_ws)
    set_default_layout(ws)

    naive = clean_naive_df(parsed07.naive_readable)
    model_summary = clean_summary_df(parsed07.model_summary)

    rows = [
        ("실험 목적", "Stage 3 next-time 성능 해석을 위한 simple baseline 비교"),
        ("비교 대상", "global mean, global median, current activity mean, prefix length mean"),
        ("main time metric", "MAE"),
        ("해석 포인트", "trivial predictor와 process-aware simple baseline 대비 multitask time head 성능 확인"),
    ]
    bullets = [
        "global_mean / prefix_len_mean은 매우 약한 baseline으로 나타났고, 실제로 강한 simple baseline은 global_median과 activity_mean이었음.",
        "Stage 3 multitask들은 대체로 trivial baseline보다 낫지만, activity_mean 같은 process-aware simple baseline을 압도하진 못함.",
        "즉 next-time 예측은 의미는 있으나, simple heuristic 대비 아주 강하다고 보긴 어려움.",
    ]
    next_row = write_summary_block(ws, "Stage 3 next-time naive baseline 분석", rows, bullets, max_col=24)

    if not naive.empty:
        test_naive = naive[naive["split"] == "test"].copy()
        write_dataframe(ws, next_row + 1, 2, test_naive, "Naive baseline (test split)")
        next_row += len(test_naive) + 4

    if not model_summary.empty:
        model_summary = model_summary.copy()
        cols = ["variant"]
        for metric in [
            "best_test_at_best_valid_task_time_mae",
            "best_test_at_best_valid_task_time_rmse",
            "best_test_at_best_valid_task_time_median_ae",
        ]:
            for stat in ("mean", "std"):
                key = f"{metric}|{stat}"
                if key in model_summary.columns:
                    cols.append(key)
        model_summary = model_summary[cols]
        write_dataframe(ws, next_row, 2, model_summary, "Stage 3 multitask time metric summary")


def ensure_backup(workbook_path: Path) -> Path:
    backup_path = workbook_path.with_name(f"{workbook_path.stem}__backup_before_stage3_{datetime.now():%Y%m%d_%H%M%S}{workbook_path.suffix}")
    copy2(workbook_path, backup_path)
    return backup_path


def main() -> None:
    parsed = {exp_no: load_notebook_tables(path) for exp_no, path in STAGE3_NOTEBOOKS.items()}

    backup_path = ensure_backup(WORKBOOK_PATH)
    wb = openpyxl.load_workbook(WORKBOOK_PATH)

    for name in NEW_SHEETS:
        if name in wb.sheetnames:
            del wb[name]

    template_ws = wb["2-2.attnbias_refine"]

    overview_ws = wb.create_sheet("3-0.Stage3")
    create_stage3_overview(overview_ws, template_ws, parsed)

    sheet_map = {
        "3-1.Baseline_mt": "02",
        "3-2.AttnBias_mt": "03",
        "3-3.AttnBias_w01": "04",
        "3-4.Anchor_w01": "05",
        "3-5.Refine_w01": "06",
    }
    for sheet_name, exp_no in sheet_map.items():
        ws = wb.create_sheet(sheet_name)
        create_stage3_experiment_sheet(ws, template_ws, exp_no, parsed[exp_no])

    naive_ws = wb.create_sheet("3-6.Time_naive")
    create_stage3_naive_sheet(naive_ws, template_ws, parsed["07"])

    wb.save(WORKBOOK_PATH)
    print(f"[ok] backup created: {backup_path}")
    print(f"[ok] workbook updated: {WORKBOOK_PATH}")
    print("[ok] added sheets:", ", ".join(NEW_SHEETS))


if __name__ == "__main__":
    main()
