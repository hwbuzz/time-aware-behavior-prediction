from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET

import pandas as pd


XES_NS = {"xes": "http://www.xes-standard.org/"}
EVENT_SORT_COLUMNS = ["case_id", "timestamp", "activity"]


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


@dataclass
class PreprocessResult:
    events: pd.DataFrame
    encoded_events: pd.DataFrame
    user_map: pd.DataFrame
    item_map: pd.DataFrame
    sasrec_interactions: pd.DataFrame
    encoding_summary: EncodingSummary


def _get_xes_value(parent: ET.Element, key: str) -> str | None:
    for child in parent:
        if child.attrib.get("key") == key:
            return child.attrib.get("value")
    return None


def _sort_and_index_events(events: pd.DataFrame) -> pd.DataFrame:
    result = events.sort_values(EVENT_SORT_COLUMNS).reset_index(drop=True)
    result["event_idx"] = result.groupby("case_id").cumcount()
    return result


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
    root = ET.parse(xes_path).getroot()

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
            if lifecycle_filter and (not lifecycle or lifecycle.lower() != lifecycle_filter.lower()):
                continue

            rows.append(
                {
                    "case_id": str(case_id),
                    "activity": f"{activity}+{lifecycle}" if include_lifecycle and lifecycle else str(activity),
                    "lifecycle": None if lifecycle is None else str(lifecycle),
                    "timestamp": timestamp,
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["case_id", "activity", "timestamp"])
    events = events.drop_duplicates(subset=["case_id", "activity", "lifecycle", "timestamp"])
    return _sort_and_index_events(events)


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

    case_starts = events.groupby("case_id", as_index=False)["timestamp"].min()

    if strategy == "earliest":
        selected_cases = case_starts.sort_values("timestamp").head(n_cases)["case_id"]
    elif strategy == "latest":
        selected_cases = case_starts.sort_values("timestamp", ascending=False).head(n_cases)["case_id"]
    elif strategy == "random":
        selected_cases = case_starts.sample(
            n=min(n_cases, len(case_starts)), random_state=random_state
        )["case_id"]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    sampled = events[events["case_id"].isin(selected_cases)].copy()
    # Full-case sampling keeps within-case order, but recomputing keeps the
    # sampled dataframe self-contained if event-level filtering is added later.
    return _sort_and_index_events(sampled)


def filter_short_cases(events: pd.DataFrame, min_case_length: int = 3) -> pd.DataFrame:
    """Drop cases shorter than the minimum sequence length needed by SASRec."""
    if min_case_length < 1:
        raise ValueError("min_case_length must be at least 1")
    if events.empty:
        return events.copy()

    case_lengths = events.groupby("case_id").size()
    keep_cases = case_lengths[case_lengths >= min_case_length].index
    return _sort_and_index_events(events[events["case_id"].isin(keep_cases)].copy())


def add_time_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add per-event time deltas for later time-aware model variants."""
    result = _sort_and_index_events(events.copy()) if not events.empty else events.copy()
    if result.empty:
        result["delta_prev_seconds"] = pd.Series(dtype="float64")
        result["delta_start_seconds"] = pd.Series(dtype="float64")
        return result

    timestamps = result.groupby("case_id")["timestamp"]
    result["delta_prev_seconds"] = timestamps.diff().dt.total_seconds().fillna(0.0)
    result["delta_start_seconds"] = (result["timestamp"] - timestamps.transform("min")).dt.total_seconds()
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
        .drop(columns="timestamp")
        .reset_index(drop=True)
    )
    user_map["user_id"] = range(1, len(user_map) + 1)

    item_map = pd.DataFrame({"activity": sorted(result["activity"].unique())})
    item_map["item_id"] = range(1, len(item_map) + 1)

    result = result.merge(user_map, on="case_id", how="left").merge(item_map, on="activity", how="left")
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

    return (
        encoded_events.sort_values(["user_id", "timestamp", "activity"])[["user_id", "item_id"]]
        .astype({"user_id": "int64", "item_id": "int64"})
        .reset_index(drop=True)
    )


def prepare_sasrec_dataset(events: pd.DataFrame, min_case_length: int = 3) -> PreprocessResult:
    """Run filtering, time features, ID encoding, and SASRec interaction building."""
    filtered = filter_short_cases(events, min_case_length=min_case_length)
    time_features = add_time_features(filtered)
    encoded, user_map, item_map, encoding_summary = encode_ids(time_features)
    interactions = build_sasrec_interactions(encoded)
    return PreprocessResult(filtered, encoded, user_map, item_map, interactions, encoding_summary)


def _save_csv(df: pd.DataFrame, output_path: str | Path, **to_csv_kwargs: object) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8", **to_csv_kwargs)
    return output_path


def save_sasrec_interactions(interactions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save interactions as whitespace-separated 'user_id item_id' rows for SASRec."""
    return _save_csv(interactions, output_path, sep=" ", header=False)


def save_sasrec_interactions_csv(interactions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the same interactions as CSV for inspection and analysis."""
    return _save_csv(interactions, output_path)


def save_event_table(events: pd.DataFrame, output_path: str | Path) -> Path:
    return _save_csv(events, output_path)


def save_mapping(mapping: pd.DataFrame, output_path: str | Path) -> Path:
    return _save_csv(mapping, output_path)


def save_preprocess_result(result: PreprocessResult, output_dir: str | Path) -> dict[str, Path]:
    """Save all preprocessing outputs used by SASRec and time-aware variants."""
    output_dir = Path(output_dir)
    return {
        "events": save_event_table(result.events, output_dir / "events_complete_only_filtered.csv"),
        "time_features": save_event_table(result.encoded_events, output_dir / "events_encoded_time_features.csv"),
        "user_map": save_mapping(result.user_map, output_dir / "user_map.csv"),
        "item_map": save_mapping(result.item_map, output_dir / "item_map.csv"),
        "sasrec_txt": save_sasrec_interactions(
            result.sasrec_interactions, output_dir / "sasrec_interactions.txt"
        ),
        "sasrec_csv": save_sasrec_interactions_csv(
            result.sasrec_interactions, output_dir / "sasrec_interactions.csv"
        ),
    }

