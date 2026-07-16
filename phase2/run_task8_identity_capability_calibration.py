"""Task 8 identity-capability calibration report.

This is intentionally lightweight: it consumes Task 7-R and Task 5A summaries,
records the predictive relationship, and emits the weighting decision used by
Task 5B-v2/new-training planning. Direction-level geometry is marked unavailable
unless future probe coefficient artifacts are added.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task7-dir", required=True)
    parser.add_argument("--task5a-summary", default="data/phase2/audits/task5a_identity_reaudit_20260713/task5a_identity_reaudit_summary.json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    task7_dir = Path(args.task7_dir)
    out_dir = Path(args.out_dir)
    calibration = read_json(task7_dir / "identity_capability_calibration.json")
    summary_rows = read_csv(task7_dir / "capability_probe_summary.csv")
    task5a = read_json(Path(args.task5a_summary))
    decision = calibration.get("decision", {})

    hidden_rows = [row for row in summary_rows if row.get("model_name") == "hidden_only_model"]
    raw_hidden_rows = [row for row in summary_rows if row.get("model_name") == "raw_hidden_joint_model"]
    metadata_hidden_rows = [row for row in summary_rows if row.get("model_name") == "family_hidden_joint_model"]
    shortcut_rows = [row for row in summary_rows if row.get("model_name") in {"raw_only_model", "family_only_model", "kmer_only_model"}]

    hidden_increment = mean([as_float(row.get("hidden_incremental_auroc_mean")) for row in hidden_rows])
    raw_hidden_increment = mean([as_float(row.get("hidden_incremental_auroc_mean")) for row in raw_hidden_rows])
    metadata_hidden_increment = mean([as_float(row.get("hidden_incremental_auroc_mean")) for row in metadata_hidden_rows])
    shortcut_best = max([as_float(row.get("test_separability_mean")) for row in shortcut_rows if as_float(row.get("test_separability_mean")) is not None] or [None])

    if decision.get("capability_probe_status") == "clean_formal_gate" and raw_hidden_increment is not None and raw_hidden_increment > 0.05:
        relation = "partial_overlap"
        identity_role = "low_weight_auxiliary"
        probe_weight_ratio = {"capability": 0.85, "identity": 0.15}
    elif decision.get("capability_probe_status") == "clean_formal_gate":
        relation = "low_or_uncertain_overlap"
        identity_role = "monitor_or_very_low_weight_auxiliary"
        probe_weight_ratio = {"capability": 0.95, "identity": 0.05}
    else:
        relation = "not_calibrated_probe_not_clean"
        identity_role = "monitor_only"
        probe_weight_ratio = {"capability": 1.0, "identity": 0.0}

    payload = {
        "created_at": now(),
        "task": "task8_identity_capability_calibration",
        "task7_dir": str(task7_dir),
        "capability_probe_status": decision.get("capability_probe_status"),
        "formal_success_allowed": bool(decision.get("formal_success_allowed")),
        "predictive_relationship": {
            "shortcut_best_mean_separability": shortcut_best,
            "hidden_incremental_auroc_mean": hidden_increment,
            "raw_plus_hidden_incremental_auroc_mean": raw_hidden_increment,
            "metadata_plus_hidden_incremental_auroc_mean": metadata_hidden_increment,
        },
        "subspace_relationship": {
            "status": "probe_direction_artifacts_unavailable",
            "required_future_artifacts": ["saved logistic coefficients", "layer-wise probe basis", "rank-basis metadata"],
        },
        "downstream_sentinel_relationship": {
            "status": "deferred_until_downstream_a_b_results_exist_for_clean_probe_candidates",
        },
        "decision": {
            "relationship_case": relation,
            "identity_role_for_training": identity_role,
            "probe_weight_ratio_for_task9": probe_weight_ratio,
            "do_not_zero_identity_before_causal_recovery": True,
        },
        "task5a_rows_available": len(task5a.get("rows", [])),
    }
    write_json(out_dir / "identity_capability_calibration.json", payload)
    (out_dir / "task8_decision.md").write_text(
        "# Task 8 Identity-Capability Calibration\n\n"
        f"- capability_probe_status: {payload['capability_probe_status']}\n"
        f"- relationship_case: {relation}\n"
        f"- identity_role_for_training: {identity_role}\n"
        f"- probe_weight_ratio_for_task9: {probe_weight_ratio}\n"
    )
    print(f"[task8] wrote calibration to {out_dir}")


if __name__ == "__main__":
    main()
