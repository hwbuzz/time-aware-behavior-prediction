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


def save_event_table(events: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

