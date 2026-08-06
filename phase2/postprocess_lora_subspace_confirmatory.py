"""Postprocess confirmatory LoRA attacker runs for Direction 3 Stage 1."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from phase2.analyze_lora_subspace_stage1 import (
    AUROC_BASELINE,
    MCC_BASELINE,
    VALIDATION_AUROC_BASELINE,
    VALIDATION_MCC_BASELINE,
    assign_group,
    effective_rank,
    module_layer,
    module_names_from_state,
    module_short_name,
    read_prediction_metrics,
    sha256_file,
    small_svd_from_factors,
    validate_orientation,
    write_csv,
    write_json,
)


DEFAULT_OUT_ROOT = Path("data/phase2/lora_subspace_targeting_20260729")
DEFAULT_TASK = "hvue_human_host_tropism"


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def load_plan(out_root: Path) -> dict[str, Any]:
    return json.loads((out_root / "confirmatory_attack_plan.json").read_text())


def refresh_plan_status(out_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    for row in plan["runs"]:
        run_id = row["run_id"]
        run_dir = out_root / "confirmatory_adapter_reruns" / run_id
        pred_dir = out_root / "confirmatory_adapter_predictions" / run_id
        results = run_dir / "eval_benchmarks.csv"
        ckpt = run_dir / "checkpoints" / DEFAULT_TASK / "best.pt"
        val = pred_dir / f"{DEFAULT_TASK}_val_predictions.csv"
        test = pred_dir / f"{DEFAULT_TASK}_test_predictions.csv"
        complete = all(path.exists() for path in (results, ckpt, val, test))
        row["status"] = "complete" if complete else "pending"
        row["results_path"] = str(results) if results.exists() else ""
        row["adapter_path"] = str(ckpt) if ckpt.exists() else ""
        row["validation_prediction_path"] = str(val) if val.exists() else ""
        row["test_prediction_path"] = str(test) if test.exists() else ""
    plan["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return plan


def write_metrics(out_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan["runs"]:
        if item["status"] != "complete":
            rows.append(
                {
                    "run_id": item["run_id"],
                    "batch_id": item["batch_id"],
                    "rank": item["rank"],
                    "learning_rate": item["learning_rate"],
                    "seed": item["seed"],
                    "status": item["status"],
                }
            )
            continue
        result = read_one_csv(Path(item["results_path"]))
        val = read_prediction_metrics(Path(item["validation_prediction_path"]))
        test = read_prediction_metrics(Path(item["test_prediction_path"]))
        group, reason = assign_group(str(item["batch_id"]), val["auroc"], val["mcc"])
        rows.append(
            {
                "run_id": item["run_id"],
                "batch_id": item["batch_id"],
                "rank": item["rank"],
                "learning_rate": item["learning_rate"],
                "seed": item["seed"],
                "status": "complete",
                "best_step": result["best_step"],
                "validation_auroc": val["auroc"],
                "validation_mcc": val["mcc"],
                "validation_auroc_excess": val["auroc"] - VALIDATION_AUROC_BASELINE,
                "validation_mcc_excess": val["mcc"] - VALIDATION_MCC_BASELINE,
                "test_auroc": test["auroc"],
                "test_mcc": test["mcc"],
                "test_auroc_excess": test["auroc"] - AUROC_BASELINE,
                "test_mcc_excess": test["mcc"] - MCC_BASELINE,
                "selected_threshold": result["validation_selected_mcc_threshold"],
                "assigned_group": group,
                "reason": reason,
                "adapter_path": item["adapter_path"],
                "adapter_sha256": sha256_file(Path(item["adapter_path"])),
                "validation_prediction_path": item["validation_prediction_path"],
                "test_prediction_path": item["test_prediction_path"],
                "global_seed": result.get("global_seed", item["seed"]),
                "seed_controls": result.get("seed_controls", "not_recorded_in_this_run"),
                "lora_init_hash": result.get("lora_init_hash", ""),
                "head_init_hash": result.get("head_init_hash", ""),
                "first_batch_ids": result.get("first_batch_ids", ""),
                "training_data_order_hash": result.get("training_data_order_hash", ""),
                "exact_command": " ".join(str(part) for part in item["command"]),
            }
        )
    write_csv(out_root / "confirmatory_adapter_metrics.csv", rows)
    return rows


def extract_confirmatory_updates(out_root: Path, metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = out_root / "confirmatory_effective_updates"
    out_dir.mkdir(exist_ok=True)
    stat_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        if row.get("status") != "complete":
            continue
        rank = int(row["rank"])
        alpha = rank * 2
        scale = alpha / rank
        payload = torch.load(row["adapter_path"], map_location="cpu")
        state = payload["state_dict"]
        for module in module_names_from_state(state):
            A = state[module + ".lora_A"].float()
            B = state[module + ".lora_B"].float()
            singular = small_svd_from_factors(A, B, scale)
            fro = float(torch.linalg.vector_norm(singular).item())
            max_abs, max_rel = validate_orientation(A, B, scale, seed=int(row["seed"]) + module_layer(module) + len(module))
            stat_rows.append(
                {
                    "run_id": row["run_id"],
                    "batch_id": row["batch_id"],
                    "assigned_group": row["assigned_group"],
                    "rank": rank,
                    "learning_rate": row["learning_rate"],
                    "seed": row["seed"],
                    "module": module,
                    "layer": module_layer(module),
                    "module_short_name": module_short_name(module),
                    "scale": scale,
                    "frobenius_norm": fro,
                    "spectral_norm": float(singular.max().item()) if singular.numel() else 0.0,
                    "effective_rank_99pct": effective_rank(singular),
                    "top_singular_values": ";".join(f"{float(x):.8g}" for x in singular[: min(16, len(singular))]),
                    "merge_equivalence_max_abs": max_abs,
                    "merge_equivalence_max_rel": max_rel,
                }
            )
            merge_rows.append(
                {
                    "run_id": row["run_id"],
                    "module": module,
                    "status": "pass" if max_abs <= 1e-4 else "fail",
                    "max_abs_diff": max_abs,
                    "max_relative_diff": max_rel,
                }
            )
    write_csv(out_dir / "confirmatory_effective_update_statistics.csv", stat_rows)
    write_csv(out_dir / "confirmatory_adapter_merge_equivalence_by_module.csv", merge_rows)
    registry = {
        "status": "complete" if stat_rows else "pending_no_completed_confirmatory_runs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": sorted({row["run_id"] for row in stat_rows}),
        "module_rows": len(stat_rows),
        "merge_equivalence_rows": len(merge_rows),
        "merge_equivalence_failures": sum(1 for row in merge_rows if row["status"] != "pass"),
        "max_merge_abs_diff": max([float(row["max_abs_diff"]) for row in merge_rows], default=None),
        "dense_update_policy": "not_saved; compact scaled low-rank statistics only",
    }
    write_json(out_dir / "confirmatory_effective_update_registry.json", registry)
    return registry


def write_summary(out_root: Path, plan: dict[str, Any], metric_rows: list[dict[str, Any]], update_registry: dict[str, Any]) -> None:
    completed = [row for row in metric_rows if row.get("status") == "complete"]
    strong = [row for row in completed if row.get("assigned_group") == "dual_metric_strong_recovery"]
    lines = [
        "# Confirmatory Run Summary",
        "",
        f"- Planned runs: `{len(plan['runs'])}`",
        f"- Completed runs: `{len(completed)}`",
        f"- Dual-metric strong completed runs: `{len(strong)}`",
        f"- Effective update rows: `{update_registry['module_rows']}`",
        f"- Merge-equivalence failures: `{update_registry['merge_equivalence_failures']}`",
        "",
        "## Completed Strong Runs",
        "",
    ]
    if strong:
        for row in strong:
            lines.append(
                f"- `{row['run_id']}`: val_ex=({float(row['validation_auroc_excess']):+.4f}, "
                f"{float(row['validation_mcc_excess']):+.4f}), test_ex=({float(row['test_auroc_excess']):+.4f}, "
                f"{float(row['test_mcc_excess']):+.4f})"
            )
    else:
        lines.append("- none yet")
    (out_root / "confirmatory_run_summary.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    out_root = args.out_dir
    plan = refresh_plan_status(out_root, load_plan(out_root))
    write_json(out_root / "confirmatory_attack_plan.json", plan)
    metric_rows = write_metrics(out_root, plan)
    update_registry = extract_confirmatory_updates(out_root, metric_rows)
    write_summary(out_root, plan, metric_rows, update_registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
