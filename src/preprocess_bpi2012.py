from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET

import pandas as pd


XES_NS = {"xes": "http://www.xes-standard.org/"}


@dataclass
class EventTableSummary:
    num_cases: int
    num_events: int
    num_activities: int
    min_case_length: int
    median_case_length: float
    max_case_length: int


@dataclass
class EncodingSummary:
    num_users: int
    num_items: int
    num_interactions: int


def _get_xes_value(parent: ET.Element, key: str) -> str | None:
    for child in parent:
        if child.attrib.get("key") == key:
            return child.attrib.get("value")
    return None


def load_bpi2012_events(
    xes_path: str | Path,
    activity_key: str = "concept:name",
    lifecycle_key: str = "lifecycle:transition",
    timestamp_key: str = "time:timestamp",
    include_lifecycle: bool = False,
    lifecycle_filter: str | None = None,
) -> pd.DataFrame:
    """Parse a BPI 2012 XES log into an event table."""
    xes_path = Path(xes_path)
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {xes_path}")

    rows: list[dict[str, object]] = []
    tree = ET.parse(xes_path)
    root = tree.getroot()

    for trace in root.findall("xes:trace", XES_NS):
        case_id = _get_xes_value(trace, "concept:name")
        if not case_id:
            continue

        for event in trace.findall("xes:event", XES_NS):
            activity = _get_xes_value(event, activity_key)
            lifecycle = _get_xes_value(event, lifecycle_key)
            timestamp = _get_xes_value(event, timestamp_key)

            if not activity or not timestamp:
                continue

            if lifecycle_filter is not None:
                if lifecycle is None or lifecycle.lower() != lifecycle_filter.lower():
                    continue

            activity_name = activity
            if include_lifecycle and lifecycle:
                activity_name = f"{activity}+{lifecycle}"

            rows.append(
                {
                    "case_id": str(case_id),
                    "activity": str(activity_name),
                    "lifecycle": None if lifecycle is None else str(lifecycle),
                    "timestamp": timestamp,
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["case_id", "activity", "timestamp"]).copy()
    events = events.drop_duplicates(
        subset=["case_id", "activity", "lifecycle", "timestamp"]
    ).copy()
    return _sort_and_index_events(events)


def _sort_and_index_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["case_id", "timestamp", "activity"]).reset_index(drop=True)
    events["event_idx"] = events.groupby("case_id").cumcount()
    return events


def summarize_event_table(events: pd.DataFrame) -> EventTableSummary:
    if events.empty:
        return EventTableSummary(0, 0, 0, 0, 0.0, 0)

    case_lengths = events.groupby("case_id").size()
    return EventTableSummary(
        num_cases=int(events["case_id"].nunique()),
        num_events=int(len(events)),
        num_activities=int(events["activity"].nunique()),
        min_case_length=int(case_lengths.min()),
        median_case_length=float(case_lengths.median()),
        max_case_length=int(case_lengths.max()),
    )


def case_length_distribution(events: pd.DataFrame) -> pd.DataFrame:
    """Return counts of cases by sequence length."""
    if events.empty:
        return pd.DataFrame(columns=["case_length", "num_cases"])

    return (
        events.groupby("case_id")
        .size()
        .value_counts()
        .rename_axis("case_length")
        .reset_index(name="num_cases")
        .sort_values("case_length")
        .reset_index(drop=True)
    )


def sample_cases(
    events: pd.DataFrame,
    n_cases: int,
    strategy: Literal["earliest", "latest", "random"] = "earliest",
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a subset of complete cases for local development."""
    if n_cases <= 0:
        raise ValueError("n_cases must be a positive integer")
    if events.empty:
        return events.copy()

    case_starts = (
        events.groupby("case_id", as_index=False)["timestamp"]
        .min()
        .rename(columns={"timestamp": "case_start"})
    )

    if strategy == "earliest":
        selected_case_ids = case_starts.sort_values("case_start").head(n_cases)["case_id"]
    elif strategy == "latest":
        selected_case_ids = case_starts.sort_values(
            "case_start", ascending=False
        ).head(n_cases)["case_id"]
    elif strategy == "random":
        selected_case_ids = case_starts.sample(
            n=min(n_cases, len(case_starts)),
            random_state=random_state,
        )["case_id"]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    sampled = events[events["case_id"].isin(selected_case_ids)].copy()
    sampled = sampled.sort_values(["case_id", "timestamp", "activity"]).reset_index(drop=True)
    # Full-case sampling preserves the original within-case event order, so this
    # is usually redundant. Recompute it anyway so the sampled dataframe remains
    # self-contained after sorting and if event-level filtering is added later.
    sampled["event_idx"] = sampled.groupby("case_id").cumcount()
    return sampled


def filter_short_cases(events: pd.DataFrame, min_case_length: int = 3) -> pd.DataFrame:
    """Drop cases shorter than the minimum sequence length needed by SASRec."""
    if min_case_length < 1:
        raise ValueError("min_case_length must be at least 1")
    if events.empty:
        return events.copy()

    case_lengths = events.groupby("case_id").size()
    keep_case_ids = case_lengths[case_lengths >= min_case_length].index
    filtered = events[events["case_id"].isin(keep_case_ids)].copy()
    return _sort_and_index_events(filtered)


def add_time_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add per-event time deltas for later time-aware model variants."""
    if events.empty:
        result = events.copy()
        result["delta_prev_seconds"] = pd.Series(dtype="float64")
        result["delta_start_seconds"] = pd.Series(dtype="float64")
        return result

    result = _sort_and_index_events(events.copy())
    grouped = result.groupby("case_id")["timestamp"]
    result["delta_prev_seconds"] = grouped.diff().dt.total_seconds().fillna(0.0)
    result["delta_start_seconds"] = (result["timestamp"] - grouped.transform("min")).dt.total_seconds()
    return result


def encode_ids(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, EncodingSummary]:
    """Encode case_id/activity into SASRec-style integer user_id/item_id values."""
    if events.empty:
        empty_users = pd.DataFrame(columns=["case_id", "user_id"])
        empty_items = pd.DataFrame(columns=["activity", "item_id"])
        return events.copy(), empty_users, empty_items, EncodingSummary(0, 0, 0)

    result = _sort_and_index_events(events.copy())

    user_map = (
        result.groupby("case_id", as_index=False)["timestamp"]
        .min()
        .sort_values(["timestamp", "case_id"])
        .reset_index(drop=True)
        .drop(columns="timestamp")
    )
    user_map["user_id"] = range(1, len(user_map) + 1)

    item_map = pd.DataFrame({"activity": sorted(result["activity"].unique())})
    item_map["item_id"] = range(1, len(item_map) + 1)

    result = result.merge(user_map, on="case_id", how="left")
    result = result.merge(item_map, on="activity", how="left")
    result = result.sort_values(["user_id", "timestamp", "activity"]).reset_index(drop=True)
    result["event_idx"] = result.groupby("user_id").cumcount()

    summary = EncodingSummary(
        num_users=int(result["user_id"].nunique()),
        num_items=int(result["item_id"].nunique()),
        num_interactions=int(len(result)),
    )
    return result, user_map, item_map, summary


def build_sasrec_interactions(encoded_events: pd.DataFrame) -> pd.DataFrame:
    """Create the two-column interaction table expected by SASRec."""
    required = {"user_id", "item_id", "timestamp", "activity"}
    missing = required.difference(encoded_events.columns)
    if missing:
        raise ValueError(f"encoded_events is missing required columns: {sorted(missing)}")

    interactions = encoded_events.sort_values(["user_id", "timestamp", "activity"])[
        ["user_id", "item_id"]
    ].copy()
    return interactions.astype({"user_id": "int64", "item_id": "int64"})


def save_sasrec_interactions(interactions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save interactions as whitespace-separated 'user_id item_id' rows for SASRec."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(output_path, sep=" ", header=False, index=False, encoding="utf-8")
    return output_path


def save_sasrec_interactions_csv(interactions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the same interactions as CSV for inspection and analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def save_event_table(events: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def save_mapping(mapping: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

