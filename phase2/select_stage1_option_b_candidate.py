"""Select the current best Stage 1 Option B initializer candidate.

The selector reads one or more sweep summary CSVs, ranks completed candidates by
test AUROC, and writes a compact JSON record that downstream orchestration can
consume.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata


def first_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source_csv"] = str(csv_path)
                rows.append(row)
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    return sorted(
        completed,
        key=lambda row: (
            first_float(row.get("test_auroc_after_ascent")) or float("-inf"),
            first_float(row.get("val_auroc_after_ascent")) or float("-inf"),
        ),
        reverse=True,
    )


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = rank_rows(rows)
    best = ranked[0] if ranked else None
    return {
        "n_rows_seen": len(rows),
        "n_completed": len(ranked),
        "best_candidate": None
        if best is None
        else {
            "config_id": best.get("config_id", ""),
            "weights_path": best.get("weights_path", ""),
            "test_auroc_after_ascent": first_float(best.get("test_auroc_after_ascent")),
            "val_auroc_after_ascent": first_float(best.get("val_auroc_after_ascent")),
            "alpha_target": first_float(best.get("alpha_target")),
            "alpha_retain": first_float(best.get("alpha_retain")),
            "elicitation_steps": int(float(best.get("elicitation_steps", 0) or 0)),
            "ascent_steps": int(float(best.get("ascent_steps", 0) or 0)),
            "readout_disruption_flag": best.get("readout_disruption_flag", ""),
            "source_csv": best.get("_source_csv", ""),
        },
        "ranked_candidates": [
            {
                "config_id": row.get("config_id", ""),
                "weights_path": row.get("weights_path", ""),
                "test_auroc_after_ascent": first_float(row.get("test_auroc_after_ascent")),
                "val_auroc_after_ascent": first_float(row.get("val_auroc_after_ascent")),
                "alpha_target": first_float(row.get("alpha_target")),
                "alpha_retain": first_float(row.get("alpha_retain")),
                "source_csv": row.get("_source_csv", ""),
            }
            for row in ranked
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        action="append",
        dest="summary_csvs",
        default=[],
        help="Repeatable summary CSV input. If omitted, defaults to known Stage 1 Option B summaries.",
    )
    parser.add_argument(
        "--out-json",
        default="data/phase2/stage1_option_b_initializer/best_candidate.json",
    )
    args = parser.parse_args()

    summary_csvs = args.summary_csvs or [
        "data/phase2/stage1_option_b_initializer/existing_runs_sweep_summary.csv",
        "data/phase2/stage1_option_b_initializer/retain_heavy_summary.csv",
        "data/phase2/stage1_option_b_initializer/longer_candidates_summary.csv",
    ]
    rows = load_rows(summary_csvs)
    report = build_report(rows)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_metadata(
        out_path.with_name(f"{out_path.stem}_metadata.json"),
        build_run_metadata(
            args=args,
            data_paths=summary_csvs,
            extra={
                "phase": "select_stage1_option_b_candidate",
                "out_json": str(out_path),
                "summary_csv_count": len(summary_csvs),
                "n_rows_seen": report["n_rows_seen"],
                "n_completed": report["n_completed"],
                "best_candidate_config_id": ""
                if report["best_candidate"] is None
                else report["best_candidate"]["config_id"],
            },
        ),
    )
    print(f"[option-b-select] wrote {out_path}")


if __name__ == "__main__":
    main()
