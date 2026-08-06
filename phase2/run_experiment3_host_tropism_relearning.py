"""Controller for Experiment 3 Host Tropism relearning suites.

This script starts after the standalone 10-arm prefilter. It does not reselect
the candidate or reinterpret prefilter metrics; eta=0.50 is frozen here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.run_metadata import file_sha256, git_info, stable_hash


EXPERIMENT_ROOT = Path("data/phase2/standalone_single_lora_intervention_20260730")
FINAL_ROOT = EXPERIMENT_ROOT / "experiment3_host_tropism_final"
TARGET_MANIFEST = Path("data/phase2/stage1_formal_target_manifests/hvue_human_host_tropism_cluster_disjoint.csv")
RETAIN_MANIFEST = Path("data/benchmarks/hvue_gue_manifest.csv")
MODEL_DIR = Path("evo-1-8k-base")
CONFIG_PATH = Path("configs/evo-1-8k-base_inference.yml")
TASK = "hvue_human_host_tropism"
SOURCE_SEED = 49
AUROC_BASELINE = 0.8554553475
MCC_BASELINE = 0.5991934876
GUE_TASKS = [
    "gue_emp_h3",
    "gue_human_tf_1",
    "gue_mouse_1",
    "gue_prom_300_notata",
    "gue_splice_reconstructed",
]
VIRAL_TASKS = [
    "virobench_all_taxon_genus",
    "virobench_all_taxon_times",
    "virobench_dna_taxon_genus",
    "virobench_dna_taxon_times",
    "virobench_rna_taxon_genus",
    "virobench_rna_taxon_times",
]
ARM_ORDER = [
    "base",
    "source_subspace_intervention_eta2",
    "random_subspace_control_eta2",
    "random_layer_control_eta2",
]
PREFILTER_ARM_ORDER = [
    "base",
    "source_subspace_intervention_eta1",
    "random_subspace_control_eta1",
    "random_layer_control_eta1",
    "source_subspace_intervention_eta2",
    "random_subspace_control_eta2",
    "random_layer_control_eta2",
    "source_subspace_intervention_eta3",
    "random_subspace_control_eta3",
    "random_layer_control_eta3",
]
SUITES = [
    {
        "suite_id": "same_family_fresh_lora",
        "trainer": "phase2/eval_benchmarks.py",
        "training_mode": "lora",
        "rank": 16,
        "lr": "5e-5",
        "seed": SOURCE_SEED + 1000,
    },
    {
        "suite_id": "cross_config_fresh_lora",
        "trainer": "phase2/eval_benchmarks.py",
        "training_mode": "lora",
        "rank": 32,
        "lr": "5e-5",
        "seed": SOURCE_SEED + 2000,
    },
    {
        "suite_id": "full_ft",
        "trainer": "phase2/eval_benchmarks_full_ft.py",
        "training_mode": "full_ft",
        "rank": 0,
        "lr": "5e-5",
        "seed": SOURCE_SEED + 3000,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def arm_checkpoint_map() -> dict[str, str]:
    registry = load_json(EXPERIMENT_ROOT / "standalone_candidate_registry.json")
    result: dict[str, str] = {"base": ""}
    for row in registry["arms"]:
        arm_id = row["arm_id"]
        if arm_id in ARM_ORDER and arm_id != "base":
            result[arm_id] = row["checkpoint_path"]
    return result


def command_display(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def append_registry_event(path: Path, key: str, event: dict[str, Any]) -> None:
    payload = load_json(path) if path.exists() else {}
    payload.setdefault(key, []).append(event)
    write_json_atomic(path, payload)


def delete_verified_artifact(
    *,
    registry_path: Path,
    path: Path,
    replacement_path: Path,
    validation_path: Path,
    reason: str,
) -> None:
    validation = load_json(validation_path)
    if validation.get("status") != "pass":
        raise RuntimeError(f"refusing to delete {path}: validation did not pass")
    if not replacement_path.exists():
        raise FileNotFoundError(f"replacement artifact missing before deleting {path}: {replacement_path}")
    if not path.exists():
        return
    freed = path.stat().st_size
    source_hash = file_sha256(path)
    replacement_hash = file_sha256(replacement_path)
    path.unlink()
    event = {
        "event": "artifact_deleted",
        "path": str(path),
        "path_sha256_before_delete": source_hash,
        "replacement_path": str(replacement_path),
        "replacement_sha256": replacement_hash,
        "validation_path": str(validation_path),
        "validation_status": validation.get("status"),
        "freed_bytes": freed,
        "reason": reason,
        "timestamp_utc": utc_now(),
    }
    append_registry_event(registry_path, "artifact_retention_events", event)


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        payload = {}
        try:
            payload = load_json(path)
        except Exception:
            pass
        pid = int(payload.get("pid", 0) or 0)
        if pid:
            try:
                os.kill(pid, 0)
            except OSError:
                path.unlink(missing_ok=True)
                return acquire_lock(path)
        raise RuntimeError(f"Controller lock exists: {path}")
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, handle)


def release_lock(path: Path) -> None:
    path.unlink(missing_ok=True)


def install_controller_signal_handlers(
    registry_path: Path,
    lock_path: Path,
) -> dict[int, Any]:
    previous_handlers: dict[int, Any] = {}
    shutdown_started = False

    def handle_signal(signum: int, _frame) -> None:
        nonlocal shutdown_started
        signal_name = signal.Signals(signum).name
        if shutdown_started:
            raise SystemExit(128 + signum)
        shutdown_started = True
        reason = f"received {signal_name}"
        try:
            registry = load_json(registry_path) if registry_path.exists() else {}
            registry["status"] = "interrupted"
            registry["failure_reason"] = reason
            registry["ended_at_utc"] = utc_now()
            write_json_atomic(registry_path, registry)
            write_final_placeholders("blocked", reason)
        finally:
            release_lock(lock_path)
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, handle_signal)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for handled_signal, previous_handler in previous_handlers.items():
        signal.signal(handled_signal, previous_handler)


def sync_prefilter_report() -> dict[str, Any]:
    runs_root = EXPERIMENT_ROOT / "standalone_prefilter_runs"
    arm_progress = []
    for arm in PREFILTER_ARM_ORDER:
        run_dir = runs_root / arm
        progress = run_dir / "downstream" / "eval_benchmarks_progress.json"
        progress_payload = load_json(progress) if progress.exists() else {}
        completed = (
            (run_dir / "integrity_smoke.json").exists()
            and (run_dir / "eval_unlearn" / "eval_ppl.json").exists()
            and (run_dir / "downstream" / "eval_benchmarks.csv").exists()
            and (run_dir / "downstream" / "eval_benchmarks_summary.json").exists()
            and progress_payload.get("status") == "complete"
            and int(progress_payload.get("completed_tasks", 0)) == 12
            and int(progress_payload.get("expected_tasks", 12)) == 12
            and (run_dir / "predictions" / f"{TASK}_val_predictions.csv").exists()
            and (run_dir / "predictions" / f"{TASK}_test_predictions.csv").exists()
        )
        arm_progress.append({"arm_id": arm, "completed": completed})
    completed_count = sum(1 for row in arm_progress if row["completed"])
    payload = {
        "generated_at_utc": utc_now(),
        "status": "complete" if completed_count == 10 else "incomplete",
        "completed_arms": completed_count,
        "total_arms": 10,
        "active_arm": "" if completed_count == 10 else "unknown",
        "arm_progress": arm_progress,
        "candidate_selection": {
            "frozen_formal_candidate": "source_subspace_intervention_eta2",
            "eta": 0.50,
            "excluded_eta_0_25": "target suppression insufficient and retain tradeoff inferior",
            "eta_0_75_status": "high_strength_non_retain_safe_ablation",
        },
        "gue_retain_status": "partial_GUE_retain_evaluation",
        "evaluated_gue_task_count": 5,
        "full_gue_retain_gate_passed": False,
    }
    write_json_atomic(EXPERIMENT_ROOT / "standalone_prefilter_report.json", payload)
    return payload


def validate_merge_audit(refresh: bool, python_bin: str, device: str) -> dict[str, Any]:
    if refresh:
        command = [
            python_bin,
            "-u",
            "phase2/audit_source_lora_merge_equivalence.py",
            "--out-dir",
            str(EXPERIMENT_ROOT),
            "--model-dir",
            str(MODEL_DIR),
            "--config-path",
            str(CONFIG_PATH),
            "--device",
            device,
            "--max-length",
            "512",
            "--batch-size",
            "2",
            "--seed",
            str(SOURCE_SEED),
            "--abs-tol",
            "0.0001",
            "--rel-tol",
            "0.0001",
        ]
        subprocess.run(command, check=True)
    audit = load_json(EXPERIMENT_ROOT / "source_lora_merge_equivalence_audit.json")
    required = {
        "parameter_level_merge_equivalence": "pass",
        "original_classification_head_equivalence": "unavailable",
        "source_update_nonzero": "pass",
        "official_merge_vs_manual_merge": "pass",
        "backbone_functional_equivalence": "pass",
        "unmerged_source_vs_base_forward_difference": "pass",
        "manual_merged_vs_base_forward_difference": "pass",
    }
    failures = [
        f"{key}={audit.get(key)!r}, expected {value!r}"
        for key, value in required.items()
        if audit.get(key) != value
    ]
    if float(audit.get("source_update_frobenius_norm", 0.0) or 0.0) <= 0.0:
        failures.append("source_update_frobenius_norm <= 0")
    if failures:
        raise RuntimeError("merge audit failed: " + "; ".join(failures))
    return audit


def write_frozen_manifest(python_bin: str) -> dict[str, Any]:
    checkpoints = arm_checkpoint_map()
    data_paths = [TARGET_MANIFEST, RETAIN_MANIFEST, CONFIG_PATH, MODEL_DIR / "model.safetensors.index.json"]
    data_hashes = {str(path): file_sha256(path) for path in data_paths if path.exists() and path.is_file()}
    for arm, ckpt in checkpoints.items():
        if ckpt:
            path = Path(ckpt)
            if not path.exists():
                raise FileNotFoundError(f"Missing frozen checkpoint for {arm}: {ckpt}")
            data_hashes[ckpt] = file_sha256(path)
    git = git_info()
    manifest = {
        "experiment_identity": "Experiment 3 - Host Tropism Standalone Single-Source LoRA Intervention",
        "generated_at_utc": utc_now(),
        "formal_candidate": {"arm_id": "source_subspace_intervention_eta2", "eta": 0.50},
        "starting_checkpoints": checkpoints,
        "target_benchmark_manifest": str(TARGET_MANIFEST),
        "retain_benchmark_manifest": str(RETAIN_MANIFEST),
        "split_type": "cluster_disjoint",
        "host_tropism_task": TASK,
        "retain_tasks": {"gue": GUE_TASKS, "viral": VIRAL_TASKS},
        "gue_retain_status": "partial_GUE_retain_evaluation",
        "evaluated_gue_task_count": 5,
        "full_gue_retain_gate_passed": False,
        "training_suites": SUITES,
        "shared_training": {
            "epochs": 3,
            "max_steps": 0,
            "eval_every": 200,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "max_length": 512,
            "metric_for_best": "auroc",
            "mcc_threshold_rule": "select threshold on validation predictions, freeze before test",
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
        "test_baselines": {"auroc": AUROC_BASELINE, "mcc": MCC_BASELINE},
        "python_bin": python_bin,
        "data_hashes": data_hashes,
        "code": git,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json_atomic(FINAL_ROOT / "frozen_protocol_manifest.json", manifest)
    write_json_atomic(
        FINAL_ROOT / "candidate_selection_report.json",
        {
            "generated_at_utc": utc_now(),
            "selection_status": "frozen_before_relearning",
            "formal_candidate": "source_subspace_intervention_eta2",
            "eta": 0.50,
            "excluded": {
                "eta_0_25": "excluded",
                "eta_0_75": "high_strength_non_retain_safe_ablation",
            },
        },
    )
    return manifest


def initialize_suite_files(suite_dir: Path, suite: dict[str, Any]) -> None:
    for dirname in ("predictions", "checkpoints", "logs", "arms"):
        (suite_dir / dirname).mkdir(parents=True, exist_ok=True)
    write_json_atomic(suite_dir / "manifest.json", suite)
    if not (suite_dir / "run_registry.json").exists():
        write_json_atomic(suite_dir / "run_registry.json", {"status": "planned", "arms": []})
    for csv_name, header in {
        "adaptation_curves.csv": ["arm_id", "step", "train_loss", "val_loss", "validation_auroc", "validation_mcc"],
        "summary.csv": [
            "arm_id",
            "status",
            "test_auroc",
            "test_mcc",
            "auroc_excess",
            "mcc_excess",
            "best_step",
            "validation_selected_mcc_threshold",
        ],
    }.items():
        path = suite_dir / csv_name
        if not path.exists():
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerow(header)
    if not (suite_dir / "thresholds.json").exists():
        write_json_atomic(suite_dir / "thresholds.json", {})
    if not (suite_dir / "report.md").exists():
        write_text_atomic(suite_dir / "report.md", f"# {suite['suite_id']}\n\nStatus: planned\n")


def suite_command(args: argparse.Namespace, suite: dict[str, Any], arm: str, checkpoints: dict[str, str]) -> list[str]:
    suite_dir = FINAL_ROOT / suite["suite_id"]
    out_dir = suite_dir / "arms" / arm
    pred_dir = suite_dir / "predictions" / arm
    export_dir = suite_dir / "checkpoints" / arm
    command = [
        args.python_bin,
        "-u",
        suite["trainer"],
        "--benchmark-manifest",
        str(TARGET_MANIFEST),
        "--benchmark-scope",
        "task",
        "--task-filter",
        TASK,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(suite["seed"]),
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
        suite["lr"],
        "--train-batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--max-length",
        "512",
        "--device",
        args.device,
        "--cpu-threads",
        "16",
        "--metric-for-best",
        "auroc",
        "--split-type",
        "cluster_disjoint",
        "--kmer-baseline-score",
        str(AUROC_BASELINE),
        "--export-predictions",
        "--prediction-dir",
        str(pred_dir),
        "--attack-recipe-id",
        f"experiment3_{suite['suite_id']}_{arm}",
        "--resume",
    ]
    if checkpoints[arm]:
        command.extend(["--ckpt", checkpoints[arm]])
    if suite["training_mode"] == "lora":
        command.extend(
            [
                "--lora-rank",
                str(suite["rank"]),
                "--lora-alpha",
                str(int(suite["rank"]) * 2),
                "--lora-dropout",
                "0.0",
            ]
        )
    else:
        command.extend(
            [
                "--export-attack-ckpt-dir",
                str(export_dir),
                "--export-attack-policy",
                "full",
            ]
        )
    return command


def adapter_head_checkpoint(suite_dir: Path, arm: str) -> Path:
    return suite_dir / "arms" / arm / "checkpoints" / TASK / "best.pt"


def canonical_full_checkpoint(suite_dir: Path, arm: str) -> Path:
    return suite_dir / "checkpoints" / arm / TASK / "weights.safetensors"


def arm_complete(suite_dir: Path, arm: str, suite: dict[str, Any] | None = None) -> bool:
    run_dir = suite_dir / "arms" / arm
    pred_dir = suite_dir / "predictions" / arm
    progress_path = run_dir / "eval_benchmarks_progress.json"
    results = run_dir / "eval_benchmarks.csv"
    if not (progress_path.exists() and results.exists()):
        return False
    if suite and suite.get("training_mode") == "lora":
        if not adapter_head_checkpoint(suite_dir, arm).exists():
            return False
    elif suite is None:
        if not (adapter_head_checkpoint(suite_dir, arm).exists() or canonical_full_checkpoint(suite_dir, arm).exists()):
            return False
    else:
        if not canonical_full_checkpoint(suite_dir, arm).exists():
            return False
    progress = load_json(progress_path)
    if progress.get("status") != "complete":
        return False
    with results.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("task") == TASK]
    return bool(rows) and (pred_dir / f"{TASK}_val_predictions.csv").exists() and (pred_dir / f"{TASK}_test_predictions.csv").exists()


def read_arm_metrics(suite_dir: Path, arm: str) -> dict[str, Any]:
    with (suite_dir / "arms" / arm / "eval_benchmarks.csv").open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row.get("task") == TASK)
    auroc = float(row["auroc"])
    mcc = float(row["mcc"])
    return {
        "arm_id": arm,
        "test_auroc": auroc,
        "test_mcc": mcc,
        "auroc_excess": auroc - AUROC_BASELINE,
        "mcc_excess": mcc - MCC_BASELINE,
        "best_step": row.get("best_step", ""),
        "validation_selected_mcc_threshold": row.get("validation_selected_mcc_threshold", ""),
        "selected_checkpoint": row.get("exported_attack_checkpoint") or str(adapter_head_checkpoint(suite_dir, arm)),
    }


def refresh_suite_summaries(suite_dir: Path) -> dict[str, Any]:
    metrics = []
    thresholds = {}
    for arm in ARM_ORDER:
        if arm_complete(suite_dir, arm):
            row = read_arm_metrics(suite_dir, arm)
            metrics.append(row)
            thresholds[arm] = row["validation_selected_mcc_threshold"]
    with (suite_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm_id",
                "status",
                "test_auroc",
                "test_mcc",
                "auroc_excess",
                "mcc_excess",
                "best_step",
                "validation_selected_mcc_threshold",
            ],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "arm_id": row["arm_id"],
                    "status": "complete",
                    "test_auroc": row["test_auroc"],
                    "test_mcc": row["test_mcc"],
                    "auroc_excess": row["auroc_excess"],
                    "mcc_excess": row["mcc_excess"],
                    "best_step": row["best_step"],
                    "validation_selected_mcc_threshold": row["validation_selected_mcc_threshold"],
                }
            )
    write_json_atomic(suite_dir / "thresholds.json", thresholds)
    return {"completed": len(metrics), "metrics": metrics}


def evaluate_base_recovery(suite_dir: Path) -> dict[str, Any]:
    base = read_arm_metrics(suite_dir, "base")
    valid = base["auroc_excess"] > 0 and base["mcc_excess"] > 0
    return {
        "status": "evaluator_valid" if valid else "evaluator_invalid",
        "suite_id": suite_dir.name,
        "base_test_auroc": base["test_auroc"],
        "base_test_mcc": base["test_mcc"],
        "base_auroc_excess": base["auroc_excess"],
        "base_mcc_excess": base["mcc_excess"],
        "reason": ""
        if valid
        else (
            "base post-adaptation recovery did not exceed both k-mer baselines; "
            "suite scientific interpretation is invalid, but independent suites can continue"
        ),
        "generated_at_utc": utc_now(),
    }


def load_smoke_gate_fields() -> dict[str, Any]:
    smoke_dir = FINAL_ROOT / "prelaunch_smoke"
    result: dict[str, Any] = {}
    for key, filename in (
        ("fresh_lora_training_mode_smoke", "fresh_lora_smoke.json"),
        ("full_ft_training_mode_smoke", "full_ft_smoke.json"),
    ):
        path = smoke_dir / filename
        if not path.exists():
            result[key] = "missing"
            continue
        try:
            payload = load_json(path)
        except Exception as exc:
            result[key] = f"unreadable: {exc}"
            continue
        result[key] = payload.get("status", "unknown")
    result["formal_workflow_release"] = (
        "pass"
        if result.get("fresh_lora_training_mode_smoke") == "pass"
        and result.get("full_ft_training_mode_smoke") == "pass"
        else "blocked"
    )
    return result


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"[{utc_now()}] command_start {command_display(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def validation_command(
    args: argparse.Namespace,
    *,
    suite: dict[str, Any],
    arm: str,
    suite_dir: Path,
    mode: str,
    output_json: Path,
    canonical_ckpt: Path | None = None,
) -> list[str]:
    command = [
        args.python_bin,
        "-u",
        "phase2/experiment3_artifact_retention.py",
        "--mode",
        mode,
        "--adapter-head-ckpt",
        str(adapter_head_checkpoint(suite_dir, arm)),
        "--output-json",
        str(output_json),
        "--results-csv",
        str(suite_dir / "arms" / arm / "eval_benchmarks.csv"),
        "--benchmark-manifest",
        str(TARGET_MANIFEST),
        "--model-dir",
        str(MODEL_DIR),
        "--config-path",
        str(CONFIG_PATH),
        "--device",
        args.device,
        "--seed",
        str(suite["seed"]),
        "--lora-rank",
        str(suite["rank"] or 16),
        "--lora-alpha",
        str((int(suite["rank"]) * 2) if suite["rank"] else 32),
        "--lora-dropout",
        "0.0",
    ]
    checkpoints = arm_checkpoint_map()
    if checkpoints.get(arm):
        command.extend(["--starting-ckpt", checkpoints[arm]])
    if canonical_ckpt is not None:
        command.extend(["--canonical-ckpt", str(canonical_ckpt)])
    return command


def materialize_lora_command(
    args: argparse.Namespace,
    *,
    suite: dict[str, Any],
    arm: str,
    suite_dir: Path,
    output_ckpt: Path,
    output_json: Path,
) -> list[str]:
    command = validation_command(
        args,
        suite=suite,
        arm=arm,
        suite_dir=suite_dir,
        mode="materialize_lora",
        output_json=output_json,
    )
    command.extend(["--output-ckpt", str(output_ckpt)])
    return command


def extract_full_ft_head_command(
    args: argparse.Namespace,
    *,
    suite: dict[str, Any],
    arm: str,
    suite_dir: Path,
    output_head: Path,
    output_json: Path,
) -> list[str]:
    command = validation_command(
        args,
        suite=suite,
        arm=arm,
        suite_dir=suite_dir,
        mode="extract_full_ft_head",
        output_json=output_json,
    )
    command.extend(["--output-head", str(output_head)])
    return command


def enforce_artifact_retention_for_arm(args: argparse.Namespace, suite: dict[str, Any], arm: str) -> None:
    suite_dir = FINAL_ROOT / suite["suite_id"]
    registry_path = FINAL_ROOT / "orchestration_registry.json"
    retention_dir = suite_dir / "artifact_retention" / arm
    retention_dir.mkdir(parents=True, exist_ok=True)
    if suite["training_mode"] == "lora":
        replacement = adapter_head_checkpoint(suite_dir, arm)
        validation_path = retention_dir / "adapter_prediction_reproduction.json"
        if not validation_path.exists() or load_json(validation_path).get("status") != "pass":
            rc = run_command(
                validation_command(
                    args,
                    suite=suite,
                    arm=arm,
                    suite_dir=suite_dir,
                    mode="validate_lora",
                    output_json=validation_path,
                ),
                retention_dir / "validate_lora.log",
            )
            if rc != 0:
                raise RuntimeError(f"LoRA artifact validation failed for {suite['suite_id']} {arm}")
        duplicate = canonical_full_checkpoint(suite_dir, arm)
        if duplicate.exists():
            delete_verified_artifact(
                registry_path=registry_path,
                path=duplicate,
                replacement_path=replacement,
                validation_path=validation_path,
                reason="fresh-LoRA policy retains adapter/head only; merged full-model export is duplicate",
            )
            meta = duplicate.with_name("meta.json")
            if meta.exists():
                delete_verified_artifact(
                    registry_path=registry_path,
                    path=meta,
                    replacement_path=replacement,
                    validation_path=validation_path,
                    reason="metadata for deleted duplicate merged LoRA full-model export",
                )
    else:
        canonical = canonical_full_checkpoint(suite_dir, arm)
        best_pt = adapter_head_checkpoint(suite_dir, arm)
        head_path = suite_dir / "checkpoints" / arm / TASK / "classification_head.pt"
        extract_json = retention_dir / "classification_head_extract.json"
        if best_pt.exists() and (not head_path.exists() or not extract_json.exists()):
            rc = run_command(
                extract_full_ft_head_command(
                    args,
                    suite=suite,
                    arm=arm,
                    suite_dir=suite_dir,
                    output_head=head_path,
                    output_json=extract_json,
                ),
                retention_dir / "extract_full_ft_head.log",
            )
            if rc != 0:
                raise RuntimeError(f"full-FT head extraction failed for {suite['suite_id']} {arm}")
        validation_path = retention_dir / "canonical_prediction_reproduction.json"
        if not validation_path.exists() or load_json(validation_path).get("status") != "pass":
            rc = run_command(
                validation_command(
                    args,
                    suite=suite,
                    arm=arm,
                    suite_dir=suite_dir,
                    mode="validate_full_ft",
                    output_json=validation_path,
                    canonical_ckpt=canonical,
                ),
                retention_dir / "validate_full_ft.log",
            )
            if rc != 0:
                raise RuntimeError(f"full-FT canonical artifact validation failed for {suite['suite_id']} {arm}")
        if best_pt.exists():
            delete_verified_artifact(
                registry_path=registry_path,
                path=best_pt,
                replacement_path=canonical,
                validation_path=validation_path,
                reason="full-FT policy retains canonical weights.safetensors and separate classification head, not duplicate best.pt",
            )


def retain_complete(retain_root: Path, suite_id: str, arm: str) -> bool:
    out_dir = retain_root / suite_id / arm
    progress = out_dir / "downstream" / "eval_benchmarks_progress.json"
    return (
        (out_dir / "eval_unlearn" / "eval_ppl.json").exists()
        and (out_dir / "downstream" / "eval_benchmarks.csv").exists()
        and progress.exists()
        and load_json(progress).get("status") == "complete"
    )


def run_post_adaptation_retain_for_arm(args: argparse.Namespace, suite: dict[str, Any], arm: str) -> None:
    retain_root = FINAL_ROOT / "post_adaptation_retain"
    retain_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        retain_root / "manifest.json",
        {
            "gue_retain_status": "partial_GUE_retain_evaluation",
            "evaluated_gue_task_count": 5,
            "full_gue_retain_gate_passed": False,
            "tasks": {"gue": GUE_TASKS, "viral": VIRAL_TASKS},
        },
    )
    status_path = retain_root / "run_registry.json"
    registry = load_json(status_path) if status_path.exists() else {"status": "running", "started_at_utc": utc_now(), "runs": []}
    write_json_atomic(status_path, registry)
    if retain_complete(retain_root, suite["suite_id"], arm):
        return
    suite_dir = FINAL_ROOT / suite["suite_id"]
    adapted_ckpt = canonical_full_checkpoint(suite_dir, arm)
    temp_ckpt = None
    if suite["training_mode"] == "lora":
        temp_dir = retain_root / "_tmp_materialized_lora" / suite["suite_id"] / arm
        temp_ckpt = temp_dir / "weights.safetensors"
        materialize_json = temp_dir / "materialize.json"
        rc = run_command(
            materialize_lora_command(
                args,
                suite=suite,
                arm=arm,
                suite_dir=suite_dir,
                output_ckpt=temp_ckpt,
                output_json=materialize_json,
            ),
            temp_dir / "materialize.log",
        )
        if rc != 0 or not temp_ckpt.exists():
            raise RuntimeError(f"temporary LoRA materialization failed for {suite['suite_id']} {arm}")
        adapted_ckpt = temp_ckpt
    if not adapted_ckpt.exists():
        raise FileNotFoundError(f"Missing adapted checkpoint for retain evaluation: {adapted_ckpt}")
    out_dir = retain_root / suite["suite_id"] / arm
    ppl_cmd = [
        args.python_bin,
        "-u",
        "phase2/eval_unlearn.py",
        "--ckpt",
        str(adapted_ckpt),
        "--out-dir",
        str(out_dir / "eval_unlearn"),
        "--model-dir",
        str(MODEL_DIR),
        "--config-path",
        str(CONFIG_PATH),
        "--device",
        args.device,
        "--batch-size",
        "4",
        "--max-length",
        "512",
        "--max-eval",
        "400",
        "--checkpoint-name",
        f"{suite['suite_id']}_{arm}",
        "--method-family",
        "experiment3_post_adaptation",
        "--seed",
        str(suite["seed"]),
    ]
    bench_cmd = [
        args.python_bin,
        "-u",
        "phase2/eval_benchmarks.py",
        "--ckpt",
        str(adapted_ckpt),
        "--benchmark-manifest",
        str(RETAIN_MANIFEST),
        "--benchmark-scope",
        "task",
        "--task-filter",
        ",".join(GUE_TASKS + VIRAL_TASKS),
        "--out-dir",
        str(out_dir / "downstream"),
        "--seed",
        str(suite["seed"]),
        "--epochs",
        "2",
        "--max-steps",
        "400",
        "--eval-every",
        "200",
        "--validation-max-rows",
        "1000",
        "--test-max-rows",
        "1000",
        "--lr",
        "0.0001",
        "--lora-rank",
        "8",
        "--lora-alpha",
        "16",
        "--lora-dropout",
        "0.0",
        "--train-batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--max-length",
        "512",
        "--device",
        args.device,
        "--cpu-threads",
        "16",
        "--discard-task-checkpoint",
        "--resume",
        "--export-predictions",
        "--prediction-dir",
        str(out_dir / "predictions"),
    ]
    for command, name in ((ppl_cmd, "ppl"), (bench_cmd, "downstream")):
        rc = run_command(command, out_dir / "logs" / f"{name}.log")
        registry["runs"].append({"suite_id": suite["suite_id"], "arm_id": arm, "phase": name, "exit_code": rc})
        write_json_atomic(status_path, registry)
        if rc != 0:
            raise RuntimeError(f"post-adaptation retain {name} failed for {suite['suite_id']} {arm}")
    if temp_ckpt and temp_ckpt.exists():
        validation_path = temp_ckpt.with_name("materialize.json")
        delete_verified_artifact(
            registry_path=FINAL_ROOT / "orchestration_registry.json",
            path=temp_ckpt,
            replacement_path=adapter_head_checkpoint(suite_dir, arm),
            validation_path=validation_path,
            reason="temporary merged LoRA checkpoint deleted after post-adaptation retain evaluation",
        )
    registry["status"] = "running"
    write_json_atomic(status_path, registry)


def cleanup_all_verified_lora_duplicates(args: argparse.Namespace) -> None:
    for suite in SUITES:
        if suite["training_mode"] != "lora":
            continue
        suite_dir = FINAL_ROOT / suite["suite_id"]
        for arm in ARM_ORDER:
            if arm_complete(suite_dir, arm, suite):
                enforce_artifact_retention_for_arm(args, suite, arm)


def require_free_disk_for_full_ft(args: argparse.Namespace) -> None:
    cleanup_all_verified_lora_duplicates(args)
    free_gb = shutil.disk_usage(FINAL_ROOT).free / (1024**3)
    event = {
        "event": "pre_full_ft_disk_observation",
        "free_disk_gb": free_gb,
        "required_free_disk_gb": None,
        "hard_gate": False,
        "timestamp_utc": utc_now(),
    }
    append_registry_event(FINAL_ROOT / "orchestration_registry.json", "artifact_retention_events", event)


def write_final_placeholders(status: str, reason: str = "") -> None:
    comparison = {"status": status, "reason": reason, "generated_at_utc": utc_now()}
    write_json_atomic(FINAL_ROOT / "experiment3_final_comparison.json", comparison)
    write_json_atomic(FINAL_ROOT / "experiment3_final_report.json", comparison)
    write_json_atomic(
        FINAL_ROOT / "experiment3_aris_protocol_audit.json",
        {
            "status": status,
            "gue_retain_status": "partial_GUE_retain_evaluation",
            "evaluated_gue_task_count": 5,
            "full_gue_retain_gate_passed": False,
            "generated_at_utc": utc_now(),
        },
    )
    with (FINAL_ROOT / "experiment3_final_metrics.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(["suite_id", "arm_id", "test_auroc", "test_mcc", "auroc_excess", "mcc_excess"])
    write_text_atomic(FINAL_ROOT / "experiment3_final_report.md", f"# Experiment 3 Final Report\n\nStatus: {status}\n\n{reason}\n")


def run_controller(args: argparse.Namespace) -> None:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = FINAL_ROOT / ".controller.lock"
    acquire_lock(lock_path)
    registry_path = FINAL_ROOT / "orchestration_registry.json"
    registry: dict[str, Any] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "current_stage": "prelaunch",
        "current_suite": "",
        "current_arm": "",
        "runs": [],
    }
    write_json_atomic(registry_path, registry)
    previous_signal_handlers = install_controller_signal_handlers(registry_path, lock_path)
    try:
        prefilter = sync_prefilter_report()
        if prefilter["status"] != "complete":
            raise RuntimeError("standalone prefilter is not complete 10/10")
        merge_audit = validate_merge_audit(not args.no_refresh_merge_audit, args.python_bin, args.device)
        manifest = write_frozen_manifest(args.python_bin)
        checkpoints = arm_checkpoint_map()
        smoke_gate = load_smoke_gate_fields()
        validation = {
            "status": "pass",
            "generated_at_utc": utc_now(),
            "prefilter_status": prefilter["status"],
            "merge_audit_generated_at_utc": merge_audit.get("generated_at_utc", ""),
            "frozen_manifest_hash": manifest["manifest_hash"],
            "free_disk_gb": shutil.disk_usage(FINAL_ROOT).free / (1024**3),
            **smoke_gate,
        }
        if smoke_gate.get("formal_workflow_release") != "pass":
            raise RuntimeError(f"prelaunch smoke gate failed: {smoke_gate}")
        write_json_atomic(FINAL_ROOT / "prelaunch_validation.json", validation)
        if args.prelaunch_only:
            registry["status"] = "prelaunch_complete"
            registry["ended_at_utc"] = utc_now()
            write_json_atomic(registry_path, registry)
            return
        for suite in SUITES:
            suite_dir = FINAL_ROOT / suite["suite_id"]
            initialize_suite_files(suite_dir, suite)
            if suite["suite_id"] == "full_ft":
                require_free_disk_for_full_ft(args)
            suite_registry = {
                "status": "running",
                "started_at_utc": utc_now(),
                "suite_id": suite["suite_id"],
                "arms": [],
                "evaluator_validity": {},
            }
            write_json_atomic(suite_dir / "run_registry.json", suite_registry)
            registry["current_stage"] = "training"
            registry["current_suite"] = suite["suite_id"]
            registry["current_arm"] = ""
            write_json_atomic(registry_path, registry)
            for arm in ARM_ORDER:
                if arm_complete(suite_dir, arm, suite):
                    enforce_artifact_retention_for_arm(args, suite, arm)
                    run_post_adaptation_retain_for_arm(args, suite, arm)
                    suite_registry["arms"].append(
                        {
                            "arm_id": arm,
                            "status": "complete",
                            "skipped_existing_valid": True,
                            "updated_at_utc": utc_now(),
                        }
                    )
                    write_json_atomic(suite_dir / "run_registry.json", suite_registry)
                    continue
                registry["current_arm"] = arm
                write_json_atomic(registry_path, registry)
                command = suite_command(args, suite, arm, checkpoints)
                log_path = suite_dir / "logs" / f"{arm}.log"
                run_record = {
                    "suite_id": suite["suite_id"],
                    "arm_id": arm,
                    "status": "running",
                    "started_at_utc": utc_now(),
                    "command": command,
                    "log_path": str(log_path),
                    "checkpoint": checkpoints.get(arm, ""),
                }
                registry["runs"].append(run_record)
                suite_registry["arms"].append(run_record)
                write_json_atomic(suite_dir / "run_registry.json", suite_registry)
                write_json_atomic(registry_path, registry)
                rc = run_command(command, log_path)
                run_record["exit_code"] = rc
                run_record["ended_at_utc"] = utc_now()
                run_record["status"] = "complete" if rc == 0 and arm_complete(suite_dir, arm, suite) else "failed"
                if run_record["status"] != "complete":
                    run_record["failure_reason"] = f"exit_code={rc}, output validation failed"
                    suite_registry["status"] = "blocked"
                    suite_registry["failure_reason"] = run_record["failure_reason"]
                    suite_registry["ended_at_utc"] = utc_now()
                    write_json_atomic(suite_dir / "run_registry.json", suite_registry)
                    write_json_atomic(registry_path, registry)
                    raise RuntimeError(f"{suite['suite_id']} {arm} failed; see {log_path}")
                write_json_atomic(suite_dir / "run_registry.json", suite_registry)
                write_json_atomic(registry_path, registry)
                enforce_artifact_retention_for_arm(args, suite, arm)
                run_post_adaptation_retain_for_arm(args, suite, arm)
                if arm == "base":
                    base_recovery = evaluate_base_recovery(suite_dir)
                    suite_registry["evaluator_validity"] = base_recovery
                    write_json_atomic(suite_dir / "evaluator_validity.json", base_recovery)
                    write_json_atomic(suite_dir / "run_registry.json", suite_registry)
            refresh_suite_summaries(suite_dir)
            suite_registry["status"] = suite_registry.get("evaluator_validity", {}).get("status", "complete")
            suite_registry["ended_at_utc"] = utc_now()
            write_json_atomic(suite_dir / "run_registry.json", suite_registry)
            write_text_atomic(
                suite_dir / "report.md",
                f"# {suite['suite_id']}\n\nStatus: {suite_registry['status']}\n",
            )
        registry["current_stage"] = "post_adaptation_retain"
        registry["current_suite"] = ""
        registry["current_arm"] = ""
        write_json_atomic(registry_path, registry)
        retain_registry_path = FINAL_ROOT / "post_adaptation_retain" / "run_registry.json"
        retain_registry = load_json(retain_registry_path) if retain_registry_path.exists() else {"runs": []}
        retain_registry["status"] = "complete"
        retain_registry["ended_at_utc"] = utc_now()
        write_json_atomic(retain_registry_path, retain_registry)
        registry["status"] = "complete"
        registry["current_stage"] = "final_report"
        registry["ended_at_utc"] = utc_now()
        write_json_atomic(registry_path, registry)
        write_final_placeholders("complete")
    except Exception as exc:
        registry["status"] = "blocked"
        registry["failure_reason"] = str(exc)
        registry["ended_at_utc"] = utc_now()
        write_json_atomic(registry_path, registry)
        if registry.get("current_stage") == "prelaunch":
            write_json_atomic(
                FINAL_ROOT / "prelaunch_validation.json",
                {"status": "failed", "reason": str(exc), "generated_at_utc": utc_now()},
            )
        write_final_placeholders("blocked", str(exc))
        raise
    finally:
        restore_signal_handlers(previous_signal_handlers)
        release_lock(lock_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prelaunch-only", action="store_true")
    parser.add_argument("--no-refresh-merge-audit", action="store_true")
    args = parser.parse_args()
    run_controller(args)


if __name__ == "__main__":
    main()
