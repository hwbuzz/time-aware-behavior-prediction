from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "delta_prev_zero_gap_nonfirst_examples_260518.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()

    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Non-first `delta_prev_seconds == 0` examples\n\n"
            "This notebook shows session examples where `event_idx > 0` and "
            "`delta_prev_seconds == 0`, so the event is not the first event but has "
            "the same timestamp as the previous event."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "\n"
            "DATA_PATH = Path('data/processed/bpi2012_complete_only/events_encoded_time_features.csv')\n"
            "df = pd.read_csv(DATA_PATH)\n"
            "print('rows:', len(df))\n"
            "print('columns:', df.columns.tolist())"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "zero_gap_nonfirst = df[(df['event_idx'] > 0) & (df['delta_prev_seconds'] == 0)].copy()\n"
            "print('non-first zero-gap count:', len(zero_gap_nonfirst))\n"
            "zero_gap_nonfirst[['case_id', 'event_idx', 'activity', 'lifecycle', 'timestamp', "
            "'delta_prev_seconds', 'delta_start_seconds']].head(10)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "sample_case_ids = zero_gap_nonfirst['case_id'].drop_duplicates().head(5).tolist()\n"
            "sample_case_ids"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "for case_id in sample_case_ids:\n"
            "    print('=' * 100)\n"
            "    print('case_id:', case_id)\n"
            "    case_df = df[df['case_id'] == case_id].copy()\n"
            "    case_df = case_df[['case_id', 'event_idx', 'activity', 'lifecycle', 'timestamp', "
            "'delta_prev_seconds', 'delta_start_seconds']]\n"
            "    display(case_df)\n"
            "    print('zero-gap non-first rows in this case:')\n"
            "    display(case_df[(case_df['event_idx'] > 0) & (case_df['delta_prev_seconds'] == 0)])"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "case_id = sample_case_ids[0]\n"
            "case_df = df[df['case_id'] == case_id].copy()\n"
            "idxs = case_df[(case_df['event_idx'] > 0) & (case_df['delta_prev_seconds'] == 0)]['event_idx'].tolist()\n"
            "print('focus case_id:', case_id)\n"
            "print('zero-gap non-first event_idx:', idxs)\n"
            "for idx in idxs:\n"
            "    print('-' * 100)\n"
            "    print('window around event_idx =', idx)\n"
            "    window = case_df[(case_df['event_idx'] >= idx - 2) & (case_df['event_idx'] <= idx + 2)].copy()\n"
            "    display(window[['case_id', 'event_idx', 'activity', 'lifecycle', 'timestamp', "
            "'delta_prev_seconds', 'delta_start_seconds']])"
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.x"}

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
