from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocess_bpi2012 import prepare_sasrec_dataset, save_preprocess_result


DEFAULT_INPUT = Path("data/interim/bpi2012_events_complete_only.csv")
DEFAULT_OUTPUT = Path("data/processed/bpi2012_complete_only_stage3_v2")


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate a clean processed dataset for Stage 3 multitask experiments."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Interim event CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Processed output directory")
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="If output-dir already exists, move it aside to a timestamped backup before regenerating.",
    )
    parser.add_argument(
        "--min-case-length",
        type=int,
        default=3,
        help="Minimum case length for SASRec preprocessing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input interim CSV not found: {args.input}")

    if args.output_dir.exists():
        if args.backup_existing:
            backup_dir = args.output_dir.with_name(
                f"{args.output_dir.name}__backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.move(str(args.output_dir), str(backup_dir))
            print(f"[info] moved existing output to backup: {backup_dir}")
        elif any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {args.output_dir}. "
                "Use --backup-existing or choose a new directory."
            )

    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    result = prepare_sasrec_dataset(raw, min_case_length=args.min_case_length, verbose=False)
    saved = save_preprocess_result(result, args.output_dir)

    encoded = result.encoded_events
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_interim_csv": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "min_case_length": args.min_case_length,
        "num_cases": int(encoded["case_id"].nunique()),
        "num_events": int(len(encoded)),
        "num_users": int(encoded["user_id"].nunique()),
        "num_items": int(encoded["item_id"].nunique()),
        "required_stage3_columns": [
            "delta_prev_seconds",
            "delta_start_seconds",
            "delta_next_seconds",
        ],
        "blank_counts": {
            "timestamp": int(encoded["timestamp"].astype(str).eq("").sum()),
            "delta_prev_seconds": int(encoded["delta_prev_seconds"].astype(str).eq("").sum()),
            "delta_start_seconds": int(encoded["delta_start_seconds"].astype(str).eq("").sum()),
            "delta_next_seconds": int(encoded["delta_next_seconds"].astype(str).eq("").sum()),
        },
        "null_counts": {
            "timestamp": int(encoded["timestamp"].isna().sum()),
            "delta_prev_seconds": int(encoded["delta_prev_seconds"].isna().sum()),
            "delta_start_seconds": int(encoded["delta_start_seconds"].isna().sum()),
            "delta_next_seconds": int(encoded["delta_next_seconds"].isna().sum()),
        },
        "files": {name: str(path.resolve()) for name, path in saved.items()},
        "sha256": {name: sha256sum(path) for name, path in saved.items()},
        "notes": [
            "This dataset is intended for Stage 3 multitask experiments.",
            "It is versioned separately to preserve the Stage 2 processed dataset.",
        ],
    }

    metadata_path = args.output_dir / "stage3_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] regenerated Stage 3 processed dataset at: {args.output_dir}")
    print(f"[ok] metadata written to: {metadata_path}")
    print(json.dumps(metadata["blank_counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
