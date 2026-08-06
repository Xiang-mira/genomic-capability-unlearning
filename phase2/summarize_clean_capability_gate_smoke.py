"""Summarize smoke-evaluated clean capability-gate candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.run_metadata import build_run_metadata, file_sha256, git_info, stable_hash, write_metadata


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def validity_passes(validity: dict[str, Any]) -> bool:
    action = validity.get("decision", {}).get("action", "")
    return bool(action) and not action.startswith("pause")


def shortcut_gap(calibration: dict[str, Any]) -> float | None:
    decision = calibration.get("decision", {})
    hidden = as_float(decision.get("hidden_mean_separability"))
    shortcut = as_float(decision.get("shortcut_best_mean_separability"))
    if hidden is None or shortcut is None:
        return None
    return shortcut - hidden


def seed_increment_summary(metrics: list[dict[str, str]], model_name: str) -> tuple[dict[str, float], int]:
    per_seed: dict[str, list[float]] = {}
    for row in metrics:
        if row.get("model_name") != model_name:
            continue
        seed = row.get("seed", "")
        value = as_float(row.get("hidden_incremental_auroc"))
        if seed and value is not None:
            per_seed.setdefault(seed, []).append(value)
    means = {seed: float(np.mean(values)) for seed, values in per_seed.items() if values}
    positive = sum(1 for value in means.values() if value > 0.0)
    return means, positive


def candidate_row(candidate: dict[str, Any], smoke_root: Path) -> dict[str, Any]:
    name = candidate["candidate_name"]
    smoke_dir = smoke_root / name
    validity = read_json(Path(candidate["candidate_dir"]) / "probe_validity" / "probe_validity_audit.json")
    calibration = read_json(smoke_dir / "identity_capability_calibration.json")
    metrics = read_csv(smoke_dir / "capability_probe_metrics.csv")
    raw_hidden = calibration.get("raw_hidden_joint_model", {})
    full_shortcut_hidden = calibration.get("raw_plus_kmer_plus_metadata_hidden_joint_model", {})
    hidden = calibration.get("hidden_only_model", {})
    raw_seed_means, raw_seed_positive = seed_increment_summary(metrics, "raw_hidden_joint_model")
    full_seed_means, full_seed_positive = seed_increment_summary(metrics, "raw_plus_kmer_plus_metadata_hidden_joint_model")
    hidden_seed_means, hidden_seed_positive = seed_increment_summary(metrics, "hidden_only_model")
    raw_hidden_increment = as_float(raw_hidden.get("hidden_incremental_auroc_mean"))
    full_shortcut_hidden_increment = as_float(full_shortcut_hidden.get("hidden_incremental_auroc_mean"))
    hidden_increment = as_float(hidden.get("hidden_incremental_auroc_mean"))
    gap = shortcut_gap(calibration)
    task7_status = calibration.get("decision", {}).get("capability_probe_status")
    passes = (
        validity_passes(validity)
        and task7_status == "clean_formal_gate"
        and full_shortcut_hidden_increment is not None
        and full_shortcut_hidden_increment >= 0.03
        and full_seed_positive >= 2
        and gap is not None
        and gap <= 0.05
    )
    return {
        "candidate_name": name,
        "candidate_quantile": candidate.get("candidate_quantile"),
        "rows": candidate.get("rows"),
        "validity_action": validity.get("decision", {}).get("action"),
        "validity_pass": validity_passes(validity),
        "task7_status": task7_status,
        "shortcut_best_mean_separability": as_float(calibration.get("decision", {}).get("shortcut_best_mean_separability")),
        "hidden_mean_separability": as_float(calibration.get("decision", {}).get("hidden_mean_separability")),
        "shortcut_minus_hidden_gap": gap,
        "hidden_only_increment_mean": hidden_increment,
        "hidden_only_positive_seeds": hidden_seed_positive,
        "raw_hidden_increment_mean": raw_hidden_increment,
        "raw_hidden_positive_seeds": raw_seed_positive,
        "full_shortcut_hidden_increment_mean": full_shortcut_hidden_increment,
        "full_shortcut_hidden_positive_seeds": full_seed_positive,
        "pass_clean_gate_smoke": passes,
        "smoke_dir": str(smoke_dir),
        "candidate_dir": candidate["candidate_dir"],
        "raw_hidden_seed_means": json.dumps(raw_seed_means, sort_keys=True),
        "full_shortcut_hidden_seed_means": json.dumps(full_seed_means, sort_keys=True),
        "hidden_seed_means": json.dumps(hidden_seed_means, sort_keys=True),
    }


def rank_key(row: dict[str, Any]) -> tuple:
    return (
        0 if row.get("pass_clean_gate_smoke") else 1,
        -(row.get("full_shortcut_hidden_increment_mean") or -999.0),
        -(row.get("full_shortcut_hidden_positive_seeds") or 0),
        row.get("shortcut_minus_hidden_gap") if row.get("shortcut_minus_hidden_gap") is not None else 999.0,
    )


def smoke_summary_signature(candidate_index: Path, smoke_root: Path) -> dict[str, Any]:
    git = git_info()
    calibration_hashes = {}
    for calibration in sorted(smoke_root.glob("*/identity_capability_calibration.json")):
        calibration_hashes[str(calibration)] = file_sha256(calibration)
    return {
        "task": "clean_gate_smoke_summary",
        "git_commit_hash": git.get("commit_hash", ""),
        "candidate_index_hash": file_sha256(candidate_index),
        "smoke_root": str(smoke_root),
        "calibration_hashes": calibration_hashes,
        "calibration_hash": stable_hash(calibration_hashes),
        "script_hash": file_sha256("phase2/summarize_clean_capability_gate_smoke.py"),
    }


def write_smoke_summary_metadata(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    signature: dict[str, Any],
    row_count: int,
    selected: dict[str, Any] | None,
) -> None:
    write_metadata(
        out_dir / "summary_metadata.json",
        build_run_metadata(
            args=args,
            source_checkpoint="clean_gate_smoke_summary",
            data_paths=[args.candidate_index, args.smoke_root],
            extra={
                "phase": "clean_gate_smoke_summary",
                "task": "clean_capability_gate_smoke_summary",
                "out_dir": str(out_dir),
                "row_count": row_count,
                "selected_candidate_name": selected.get("candidate_name") if selected else None,
                "run_signature": signature,
                "outputs": [
                    "clean_gate_smoke_summary.csv",
                    "clean_gate_smoke_summary.json",
                    "clean_gate_smoke_summary_signature.json",
                    "clean_gate_smoke_summary.md",
                ],
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    candidate_index = read_json(Path(args.candidate_index))
    smoke_root = Path(args.smoke_root)
    out_dir = Path(args.out_dir)
    signature = smoke_summary_signature(Path(args.candidate_index), smoke_root)
    rows = [candidate_row(candidate, smoke_root) for candidate in candidate_index.get("candidates", [])]
    rows.sort(key=rank_key)
    write_csv(out_dir / "clean_gate_smoke_summary.csv", rows)

    selected = next((row for row in rows if row.get("pass_clean_gate_smoke")), None)
    payload = {
        "created_at": now(),
        "task": "clean_capability_gate_smoke_summary",
        "candidate_index": args.candidate_index,
        "smoke_root": args.smoke_root,
        "rows": rows,
        "selected_candidate": selected,
        "decision": {
            "status": "go_clean_gate" if selected else "stop_no_clean_gate",
            "selected_candidate_name": selected.get("candidate_name") if selected else None,
            "reason": (
                "matched candidate passed validity plus positive incremental smoke criteria"
                if selected
                else "no candidate met validity plus positive incremental smoke criteria"
            ),
        },
        "run_signature": signature,
    }
    write_json(out_dir / "clean_gate_smoke_summary.json", payload)
    write_json(out_dir / "clean_gate_smoke_summary_signature.json", signature)
    write_smoke_summary_metadata(
        args=args,
        out_dir=out_dir,
        signature=signature,
        row_count=len(rows),
        selected=selected,
    )
    lines = [
        "# Clean Capability Gate Smoke Summary",
        "",
        f"- decision_status: {payload['decision']['status']}",
        f"- selected_candidate: {payload['decision']['selected_candidate_name']}",
        f"- reason: {payload['decision']['reason']}",
        "",
        "## Candidates",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['candidate_name']}: pass={row['pass_clean_gate_smoke']} "
            f"full_shortcut_hidden_increment_mean={row['full_shortcut_hidden_increment_mean']} "
            f"full_shortcut_hidden_positive_seeds={row['full_shortcut_hidden_positive_seeds']} "
            f"shortcut_gap={row['shortcut_minus_hidden_gap']} "
            f"validity_action={row['validity_action']} "
            f"task7_status={row['task7_status']}"
        )
    (out_dir / "clean_gate_smoke_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[clean-smoke-summary] wrote summary to {out_dir}")


if __name__ == "__main__":
    main()
