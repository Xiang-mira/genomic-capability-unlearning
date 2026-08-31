"""Formal Direction 3 LoRA-subspace targeting experiment scaffold.

The first formal gate is an artifact inventory. Existing Stage 1 calibration
runs discarded their temporary checkpoints, so this module records the missing
adapters/predictions and writes a minimum targeted rerun plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from phase2.project_python import project_python_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_ROOT = PROJECT_ROOT / "data/phase2/stage1_formal_experiment_20260727"
DEFAULT_ALIGNMENT_ROOT = PROJECT_ROOT / "data/phase2/stage1_baseline_alignment_20260729"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/lora_subspace_targeting_20260729"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv"
DEFAULT_TASK = "hvue_human_host_tropism"
STRONG_MATCHED_INPUT_KMER_AUROC = 0.8554553475149496
STRONG_MATCHED_INPUT_KMER_MCC = 0.5991934875548052
FULL_SEQUENCE_STRONG_AUROC = 0.8930006862072345
DEFAULT_STAGE1_PYTHON = project_python_path()


@dataclass(frozen=True)
class SelectedConfig:
    label: str
    rank: int
    lr_label: str
    seeds: tuple[int, ...]
    reason: str


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def format_lr_label(value: object) -> str:
    value_f = float(value)
    if value_f == 1e-5:
        return "1e-5"
    if value_f == 5e-5:
        return "5e-5"
    if value_f == 1e-4:
        return "1e-4"
    return f"{value_f:g}"


def normalize_split(value: object) -> str:
    split = str(value or "").strip().lower()
    return "val" if split in {"dev", "valid", "validation"} else split


def normalize_split_type(value: object) -> str:
    split_type = str(value or "").strip().lower()
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type or "random"


def stage1_run_dir(stage1_root: Path, rank: int, lr_label: str, seed: int) -> Path:
    return stage1_root / f"fresh_lora/base/rank_{rank}/lr_{lr_label}/seed_{seed}"


def path_hash_or_empty(path: Path) -> str:
    return file_sha256(path) if path.exists() and path.is_file() else ""


def load_json_or_empty(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_random_label_manifest(
    source_manifest: Path,
    out_manifest: Path,
    *,
    task: str = DEFAULT_TASK,
    split_type: str = "cluster_disjoint",
    seed: int = 1042,
) -> dict[str, object]:
    """Write a deterministic matched random-label manifest.

    Labels are permuted within each split for the formal task, preserving sample
    IDs, split assignments, sequences, and per-split label counts.
    """
    rows = read_csv_rows(source_manifest)
    fieldnames = list(rows[0].keys()) if rows else []
    out_rows = [dict(row) for row in rows]
    indices_by_split: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(out_rows):
        if row.get("task") != task:
            continue
        if normalize_split_type(row.get("split_type", "")) != split_type:
            continue
        indices_by_split[normalize_split(row.get("split", ""))].append(idx)

    rng = random.Random(seed)
    permutation_summary: dict[str, object] = {}
    for split, indices in sorted(indices_by_split.items()):
        labels = [out_rows[idx]["label"] for idx in indices]
        shuffled = labels[:]
        if len(set(labels)) > 1:
            for _attempt in range(100):
                rng.shuffle(shuffled)
                if shuffled != labels:
                    break
        for idx, label in zip(indices, shuffled):
            out_rows[idx]["label"] = label
        permutation_summary[split] = {
            "n": len(indices),
            "original_label_counts": {label: labels.count(label) for label in sorted(set(labels))},
            "randomized_label_counts": {label: shuffled.count(label) for label in sorted(set(shuffled))},
            "changed_count": sum(old != new for old, new in zip(labels, shuffled)),
        }

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "random_label_manifest": str(out_manifest),
        "random_label_manifest_sha256": file_sha256(out_manifest),
        "task": task,
        "split_type": split_type,
        "seed": seed,
        "policy": "permute labels within each split; preserve sample IDs, sequences, split assignments, and label marginals",
        "permutation_summary": permutation_summary,
    }


def inventory_stage1_runs(stage1_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for results_path in sorted(stage1_root.glob("fresh_lora/base/rank_*/lr_*/seed_*/eval_benchmarks.csv")):
        run_dir = results_path.parent
        rank = int(run_dir.parts[-3].split("_", 1)[1])
        lr_label = run_dir.parts[-2].split("_", 1)[1]
        seed = int(run_dir.parts[-1].split("_", 1)[1])
        run_id = f"fresh_lora_base_r{rank}_lr{lr_label}_seed{seed}"
        result = read_csv_rows(results_path)[0]
        metadata_path = run_dir / "eval_benchmarks_metadata.json"
        metadata = load_json_or_empty(metadata_path)
        progress = load_json_or_empty(run_dir / "eval_benchmarks_progress.json")
        best_checkpoint = Path(str(result.get("best_checkpoint", ""))) if result.get("best_checkpoint") else run_dir / "checkpoints" / DEFAULT_TASK / "best.pt"
        validation_prediction_path = Path(str(result.get("validation_prediction_path", ""))) if result.get("validation_prediction_path") else run_dir / "predictions" / f"{DEFAULT_TASK}_val_predictions.csv"
        test_prediction_path = Path(str(result.get("test_prediction_path", ""))) if result.get("test_prediction_path") else run_dir / "predictions" / f"{DEFAULT_TASK}_test_predictions.csv"
        checkpoint_retained = str(result.get("checkpoint_retained", "")).lower() == "true"
        row = {
            "run_id": run_id,
            "rank": rank,
            "learning_rate": float(lr_label),
            "lr_label": lr_label,
            "seed": seed,
            "lora_alpha": result.get("lora_alpha", ""),
            "scaling_rule": "lora_alpha / rank",
            "target_modules": "all Linear modules under every Evo block",
            "classification_head_configuration": "fresh linear head over mean pooled final normalized Evo states",
            "training_budget": "epochs=3,max_steps=0,eval_every=200,patience=3",
            "raw_auroc": result.get("auroc", ""),
            "raw_mcc": result.get("mcc", ""),
            "auroc_excess_over_strong_matched_kmer": "" if not result.get("auroc") else float(result["auroc"]) - STRONG_MATCHED_INPUT_KMER_AUROC,
            "gap_relative_to_full_sequence_strong": "" if not result.get("auroc") else float(result["auroc"]) - FULL_SEQUENCE_STRONG_AUROC,
            "adapter_path": str(best_checkpoint) if best_checkpoint.exists() else "",
            "adapter_hash": path_hash_or_empty(best_checkpoint),
            "classification_head_path": str(best_checkpoint) if best_checkpoint.exists() else "",
            "classification_head_hash": path_hash_or_empty(best_checkpoint),
            "validation_prediction_path": str(validation_prediction_path) if validation_prediction_path.exists() else "",
            "validation_prediction_hash": path_hash_or_empty(validation_prediction_path),
            "test_prediction_path": str(test_prediction_path) if test_prediction_path.exists() else "",
            "test_prediction_hash": path_hash_or_empty(test_prediction_path),
            "manifest_hash": metadata.get("data_hashes", {}).get("data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv", ""),
            "split_hash": metadata.get("data_hashes", {}).get("data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv", ""),
            "checkpoint_discard_status": "discarded" if not checkpoint_retained or not best_checkpoint.exists() else "retained",
            "lora_a_matrices_present": best_checkpoint.exists(),
            "lora_b_matrices_present": best_checkpoint.exists(),
            "adapter_configuration_present": metadata_path.exists(),
            "classification_head_present": best_checkpoint.exists(),
            "validation_probabilities_present": validation_prediction_path.exists(),
            "test_probabilities_present": test_prediction_path.exists(),
            "results_path": str(results_path),
            "metadata_path": str(metadata_path) if metadata_path.exists() else "",
            "progress_status": progress.get("status", ""),
        }
        rows.append(row)
    return rows


def select_targeted_configs(alignment_root: Path) -> list[SelectedConfig]:
    summary_path = alignment_root / "stage1_baseline_alignment_report.json"
    report = json.loads(summary_path.read_text())
    stable = report["conclusion"].get("stable_positive_auroc_configurations", [])
    selected: list[SelectedConfig] = []
    if stable:
        most_stable = min(stable, key=lambda row: float(row["auroc_std"]))
        selected.append(
            SelectedConfig(
                label="most_stable_positive",
                rank=int(most_stable["rank"]),
                lr_label=format_lr_label(most_stable["lr"]),
                seeds=(42, 43, 44),
                reason="lowest AUROC standard deviation among stable positive-excess configurations",
            )
        )
    selected.extend(
        [
            SelectedConfig("best_run_configuration", 32, "5e-5", (42, 43, 44), "current best individual run configuration"),
            SelectedConfig("frozen_exploratory_configuration", 16, "5e-5", (42, 43, 44), "frozen exploratory configuration"),
            SelectedConfig("weak_near_parity_configuration", 16, "1e-5", (42, 43, 44), "weak or near-parity LoRA comparison"),
        ]
    )
    dedup: dict[tuple[int, str], SelectedConfig] = {}
    for item in selected:
        dedup.setdefault((item.rank, item.lr_label), item)
    return list(dedup.values())


def command_for_rerun(
    config: SelectedConfig,
    seed: int,
    out_root: Path,
    *,
    manifest: Path = DEFAULT_MANIFEST,
    run_label: str | None = None,
) -> list[str]:
    run_name = run_label or f"{config.label}_r{config.rank}_lr{config.lr_label}_seed{seed}"
    out_dir = out_root / "selected_adapter_reruns" / run_name
    pred_dir = out_root / "selected_adapter_predictions" / run_name
    python_executable = str(DEFAULT_STAGE1_PYTHON if DEFAULT_STAGE1_PYTHON.exists() else Path(sys.executable))
    return [
        python_executable,
        "-u",
        "phase2/eval_benchmarks.py",
        "--benchmark-manifest",
        str(manifest),
        "--benchmark-scope",
        "task",
        "--task-filter",
        DEFAULT_TASK,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--epochs",
        "3",
        "--max-steps",
        "0",
        "--eval-every",
        "200",
        "--validation-max-rows",
        "0",
        "--test-max-rows",
        "0",
        "--lr",
        config.lr_label,
        "--lora-rank",
        str(config.rank),
        "--lora-alpha",
        str(config.rank * 2),
        "--lora-dropout",
        "0.0",
        "--train-batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--max-length",
        "512",
        "--device",
        "cuda:0",
        "--cpu-threads",
        "16",
        "--metric-for-best",
        "auroc",
        "--split-type",
        "cluster_disjoint",
        "--kmer-baseline-score",
        str(STRONG_MATCHED_INPUT_KMER_AUROC),
        "--export-predictions",
        "--prediction-dir",
        str(pred_dir),
    ]


def inspect_planned_rerun_status(out_root: Path, run_id: str) -> dict[str, object]:
    run_dir = out_root / "selected_adapter_reruns" / run_id
    pred_dir = out_root / "selected_adapter_predictions" / run_id
    progress = load_json_or_empty(run_dir / "eval_benchmarks_progress.json")
    results_path = run_dir / "eval_benchmarks.csv"
    adapter_path = run_dir / "checkpoints" / DEFAULT_TASK / "best.pt"
    val_predictions = pred_dir / f"{DEFAULT_TASK}_val_predictions.csv"
    test_predictions = pred_dir / f"{DEFAULT_TASK}_test_predictions.csv"
    complete = all(path.exists() for path in (results_path, adapter_path, val_predictions, test_predictions))
    if complete:
        status = "complete"
    elif progress.get("status") == "failed":
        status = "failed"
    elif progress.get("status") in {"running", "starting"}:
        status = "running_or_partial"
    else:
        status = "planned_not_started"
    return {
        "status": status,
        "results_path": str(results_path) if results_path.exists() else "",
        "adapter_path": str(adapter_path) if adapter_path.exists() else "",
        "validation_prediction_path": str(val_predictions) if val_predictions.exists() else "",
        "test_prediction_path": str(test_predictions) if test_predictions.exists() else "",
        "progress_status": progress.get("status", ""),
    }


def build_rerun_plan(
    inventory: list[dict[str, object]],
    alignment_root: Path,
    out_root: Path,
    random_label_manifest: Path | None = None,
) -> dict[str, object]:
    selected_configs = select_targeted_configs(alignment_root)
    planned = []
    for config in selected_configs:
        for seed in config.seeds:
            planned.append(
                {
                    "run_id": f"{config.label}_r{config.rank}_lr{config.lr_label}_seed{seed}",
                    "selection_label": config.label,
                    "rank": config.rank,
                    "learning_rate": float(config.lr_label),
                    "seed": seed,
                    "reason": config.reason,
                    "command": command_for_rerun(config, seed, out_root),
                    **inspect_planned_rerun_status(out_root, f"{config.label}_r{config.rank}_lr{config.lr_label}_seed{seed}"),
                }
            )
    control_config = selected_configs[0] if selected_configs else SelectedConfig("most_stable_positive", 16, "5e-5", (42, 43, 44), "fallback")
    for seed in control_config.seeds:
        run_id = f"matched_random_label_control_r{control_config.rank}_lr{control_config.lr_label}_seed{seed}"
        planned.append(
            {
                "run_id": run_id,
                "selection_label": "matched_random_label_control",
                "rank": control_config.rank,
                "learning_rate": float(control_config.lr_label),
                "seed": seed,
                "reason": "matched random-label control using the most stable positive configuration geometry",
                "command": []
                if random_label_manifest is None
                else command_for_rerun(
                    SelectedConfig("matched_random_label_control", control_config.rank, control_config.lr_label, (seed,), "control"),
                    seed,
                    out_root,
                    manifest=random_label_manifest,
                    run_label=run_id,
                ),
                **inspect_planned_rerun_status(out_root, run_id),
            }
        )
    missing_counts = {
        "adapter_missing": sum(not bool(row["adapter_path"]) for row in inventory),
        "validation_predictions_missing": sum(not bool(row["validation_prediction_path"]) for row in inventory),
        "test_predictions_missing": sum(not bool(row["test_prediction_path"]) for row in inventory),
    }
    return {
        "decision": "execute_task2_minimum_targeted_adapter_reruns",
        "reason": "Existing 27-run Stage 1 products discarded task checkpoints and did not export validation/test probabilities.",
        "missing_counts": missing_counts,
        "selected_configs": [config.__dict__ for config in selected_configs],
        "planned_reruns": planned,
        "completion_gate": {
            "normal_label_adapters_one_common_rank_lr_min": 3,
            "adapters_from_two_ranks_required_where_available": True,
            "weak_or_random_label_control_required": True,
            "complete_scaling_and_target_module_metadata_required": True,
            "current_status": "not_satisfied",
        },
        "random_label_manifest": "" if random_label_manifest is None else str(random_label_manifest),
    }


def write_placeholder_deliverables(out_root: Path, status: str) -> None:
    empty_csvs = {
        "formal_mcc_alignment.csv": ["run_id", "status", "reason"],
        "effective_update_statistics.csv": ["run_id", "module", "status", "reason"],
        "subspace_pairwise_metrics.csv": ["run_a", "run_b", "metric", "value", "status"],
        "fresh_attack_curves.csv": ["checkpoint", "attack", "step", "auroc", "mcc", "status"],
    }
    for name, fields in empty_csvs.items():
        row = {field: "" for field in fields}
        if "status" in row:
            row["status"] = status
        if "reason" in row:
            row["reason"] = "blocked before this stage by missing adapter reruns"
        write_csv(out_root / name, [row], fields)
    json_placeholders = {
        "effective_update_registry.json": {"status": status},
        "subspace_stability_report.json": {"status": status},
        "consensus_subspace_registry.json": {"status": status},
        "consensus_go_no_go_report.json": {"status": status, "decision": "not_reached"},
        "candidate_checkpoint_registry.json": {"status": status, "candidate_checkpoints": []},
        "candidate_prefilter_report.json": {"status": status},
        "fresh_attack_evaluation.json": {"status": status},
    }
    for name, payload in json_placeholders.items():
        write_json(out_root / name, payload)
    md_placeholders = {
        "effective_update_statistics.md": "# Effective Update Statistics\n\nBlocked until targeted adapter reruns produce retained adapter checkpoints.\n",
        "subspace_stability_report.md": "# Subspace Stability Report\n\nNot reached. Adapter artifacts and controls are missing.\n",
        "consensus_go_no_go_report.md": "# Consensus Go/No-Go Report\n\nDecision not reached. Do not generate edited checkpoints until Task 2 and update extraction pass.\n",
        "candidate_prefilter_report.md": "# Candidate Prefilter Report\n\nNot reached. No candidate checkpoints were generated.\n",
        "fresh_attack_evaluation.md": "# Fresh Attack Evaluation\n\nNot reached. Held-out attacks must wait for a consensus Go decision and retained candidate checkpoints.\n",
    }
    for name, text in md_placeholders.items():
        (out_root / name).write_text(text)


def run(args: argparse.Namespace) -> None:
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    for dirname in (
        "manifests",
        "selected_adapter_reruns",
        "selected_adapter_predictions",
        "effective_updates",
        "subspace_figures",
        "consensus_subspaces",
        "candidate_checkpoints",
        "fresh_attack_predictions",
    ):
        (out_root / dirname).mkdir(exist_ok=True)
    random_label_manifest = out_root / "manifests" / "hvue_human_host_tropism_random_labels_seed1042.csv"
    random_label_manifest_meta = build_random_label_manifest(
        args.manifest,
        random_label_manifest,
        task=DEFAULT_TASK,
        split_type="cluster_disjoint",
        seed=args.random_label_seed,
    )
    write_json(out_root / "manifests" / "hvue_human_host_tropism_random_labels_seed1042.json", random_label_manifest_meta)
    inventory = inventory_stage1_runs(args.stage1_root)
    write_csv(out_root / "adapter_inventory.csv", inventory)
    missing_adapters = sum(not bool(row["adapter_path"]) for row in inventory)
    missing_val = sum(not bool(row["validation_prediction_path"]) for row in inventory)
    missing_test = sum(not bool(row["test_prediction_path"]) for row in inventory)
    (out_root / "adapter_inventory.md").write_text(
        "\n".join(
            [
                "# Adapter Inventory",
                "",
                f"- Inventoried Stage 1 runs: `{len(inventory)}`",
                f"- Missing retained adapter checkpoints: `{missing_adapters}`",
                f"- Missing validation prediction tables: `{missing_val}`",
                f"- Missing test prediction tables: `{missing_test}`",
                "- Decision: execute the minimum targeted adapter reruns before effective-update extraction.",
            ]
        )
        + "\n"
    )
    rerun_plan = build_rerun_plan(inventory, args.alignment_root, out_root, random_label_manifest=random_label_manifest)
    write_json(out_root / "missing_artifacts_rerun_plan.json", rerun_plan)
    write_json(out_root / "selected_adapter_rerun_registry.json", rerun_plan)
    write_placeholder_deliverables(out_root, "not_reached_missing_targeted_adapter_reruns")
    final_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_at_task2_targeted_adapter_reruns_not_started",
        "classification": "not_yet_classifiable",
        "primary_baselines": {
            "strong_matched_input_kmer_auroc": STRONG_MATCHED_INPUT_KMER_AUROC,
            "strong_matched_input_kmer_mcc": STRONG_MATCHED_INPUT_KMER_MCC,
            "full_sequence_strong_reference_auroc": FULL_SEQUENCE_STRONG_AUROC,
        },
        "inventory_summary": {
            "stage1_runs": len(inventory),
            "missing_retained_adapters": missing_adapters,
            "missing_validation_predictions": missing_val,
            "missing_test_predictions": missing_test,
        },
        "next_gate": "Run selected_adapter_reruns with checkpoint retention and prediction export, plus a matched random-label control.",
        "random_label_manifest": random_label_manifest_meta,
        "go_no_go": "not_reached",
        "final_question_answer": "Not yet answerable. Existing artifacts do not contain the retained LoRA adapters or validation/test probability exports required to test subspace stability or causal held-out resistance.",
    }
    write_json(out_root / "final_lora_subspace_targeting_report.json", final_payload)
    (out_root / "final_lora_subspace_targeting_report.md").write_text(
        "\n".join(
            [
                "# Formal LoRA-Subspace Targeting Report",
                "",
                "## Current Status",
                "",
                "The experiment is gated at Task 2. The existing 27 Stage 1 LoRA calibration runs provide aggregate AUROC/MCC values, but they do not provide retained adapter checkpoints or validation/test probability exports.",
                "",
                "## Decision",
                "",
                "Do not construct subspaces or edited checkpoints yet. Execute the targeted rerun plan in `missing_artifacts_rerun_plan.json` first.",
                "",
                "## Final Question",
                "",
                "Whether Host Tropism relearning relies on a stable and causally useful LoRA weight subspace is not yet answerable from the current artifacts.",
            ]
        )
        + "\n"
    )
    registry = {
        "repository_commit": git_text(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_text(["status", "--porcelain"])),
        "git_status_short": git_text(["status", "--short"]).splitlines(),
        "dirty_diff_sha256": hashlib.sha256(git_text(["diff"]).encode("utf-8")).hexdigest(),
        "stage1_root": str(args.stage1_root),
        "alignment_root": str(args.alignment_root),
        "manifest": str(args.manifest),
        "random_label_manifest": str(random_label_manifest),
        "commands": [" ".join(sys.argv)],
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "artifact_hashes": {
            "adapter_inventory.csv": file_sha256(out_root / "adapter_inventory.csv"),
            "missing_artifacts_rerun_plan.json": file_sha256(out_root / "missing_artifacts_rerun_plan.json"),
            "random_label_manifest.csv": file_sha256(random_label_manifest),
        },
    }
    write_json(out_root / "experiment_registry.json", registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--random-label-seed", type=int, default=1042)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
