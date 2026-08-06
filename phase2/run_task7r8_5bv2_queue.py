"""Gated sequential queue for Task 7-R -> Task 8 -> Task 5B-v2.

The queue is meant to run unattended inside screen. It stops at hard gates
instead of continuing into expensive or scientifically invalid stages.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.run_metadata import build_run_metadata, stable_hash, write_metadata
from phase2.run_task5a_identity_reaudit import TASK3_CONTEXT


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def queue_metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.task5b_v2_out_dir) / "task7r8_5bv2_queue_metadata.json"


def task5b_v2_manifest_metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.task5b_v2_out_dir) / "task5b_v2_checkpoint_manifest_metadata.json"


def write_queue_metadata(args: argparse.Namespace) -> None:
    write_metadata(
        queue_metadata_path(args),
        build_run_metadata(
            args=args,
            source_checkpoint=args.model_dir,
            data_paths=[
                args.task5a_manifest,
                Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json",
                args.benchmark_manifest,
                args.config_path,
                Path(args.task7r_out_dir) / "task7r_internal_target_config.json",
                Path(args.task7r_out_dir) / "capability_dataset_manifest.csv",
                Path(args.task7r_out_dir) / "capability_dataset_audit.json",
                Path(args.task7r_out_dir) / "identity_capability_calibration.json",
            ],
            extra={
                "phase": "task7r8_5bv2_queue",
                "task": "task7r8_5bv2_queue",
                "task3_context": TASK3_CONTEXT,
                "task7r_out_dir": args.task7r_out_dir,
                "task8_out_dir": args.task8_out_dir,
                "task5b_v2_out_dir": args.task5b_v2_out_dir,
                "queue_status_path": args.queue_status,
                "primary_task": args.primary_task,
                "aux_task": args.aux_task,
                "formal_split_column": args.formal_split_column,
                "max_per_split_label": args.max_per_split_label,
                "n_bootstrap": args.n_bootstrap,
                "layers": args.layers,
                "batch_size": args.batch_size,
                "device": args.device,
                "checkpoint_format": args.checkpoint_format,
                "probe_seeds": args.probe_seeds,
                "fresh_c_grid": args.fresh_c_grid,
                "stop_on_low_disk_gb": args.stop_on_low_disk_gb,
            },
        ),
    )


def write_task5b_v2_manifest_metadata(args: argparse.Namespace, manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    checkpoint_names = [str(entry.get("checkpoint_name", "")) for entry in entries]
    source_names = [str(entry.get("source_checkpoint_name", entry.get("checkpoint_name", ""))) for entry in entries]
    write_metadata(
        task5b_v2_manifest_metadata_path(args),
        build_run_metadata(
            args=args,
            source_checkpoint="task5b_v2_checkpoint_manifest_builder",
            data_paths=[args.task5a_manifest, manifest_path],
            extra={
                "phase": "task5b_v2_checkpoint_manifest",
                "task": "task5b_v2_clean_probe_checkpoint_manifest",
                "manifest_path": str(manifest_path),
                "source_task5a_manifest": args.task5a_manifest,
                "checkpoint_count": len(entries),
                "checkpoint_names": checkpoint_names,
                "source_checkpoint_names": source_names,
                "checkpoint_name_hash": stable_hash(checkpoint_names),
                "source_checkpoint_name_hash": stable_hash(source_names),
            },
        ),
    )


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def write_status(args: argparse.Namespace, status: str, **extra: Any) -> None:
    write_json(
        Path(args.queue_status),
        {
            "updated_at": now(),
            "status": status,
            "task3_context": TASK3_CONTEXT,
            "task7r_out_dir": args.task7r_out_dir,
            "task8_out_dir": args.task8_out_dir,
            "task5b_v2_out_dir": args.task5b_v2_out_dir,
            "queue_metadata_path": str(queue_metadata_path(args)),
            **extra,
        },
    )


def require_disk(args: argparse.Namespace, stage: str) -> None:
    free = free_disk_gb(Path(args.task7r_out_dir))
    if free < args.stop_on_low_disk_gb:
        write_status(args, "stopped_low_disk", stage=stage, free_disk_gb=free)
        raise RuntimeError(f"low disk before {stage}: free={free:.2f}G threshold={args.stop_on_low_disk_gb:.2f}G")
    print(f"[queue7r] disk ok before {stage}: free={free:.2f}G", flush=True)


def run_command(args: argparse.Namespace, command: list[str], stage: str) -> None:
    print(f"[queue7r] start {stage}: {' '.join(command)}", flush=True)
    write_status(args, "running", stage=stage, command=command)
    started = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - started
    print(f"[queue7r] finish {stage}: returncode={result.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    if result.returncode != 0:
        write_status(args, "failed", stage=stage, returncode=result.returncode, elapsed_sec=elapsed)
        raise RuntimeError(f"stage failed: {stage} returncode={result.returncode}")


def build_task5b_v2_manifest(args: argparse.Namespace) -> Path:
    source = read_json(Path(args.task5a_manifest))
    by_name = {row.get("checkpoint_name"): row for row in source.get("checkpoints", [])}
    by_source = {row.get("source_checkpoint_name", row.get("checkpoint_name")): row for row in source.get("checkpoints", [])}
    entries = []
    for name in ["base", "projection_rank32", "best_rmu_from_task5a", "best_gd_from_task5a", "gd_random_control"]:
        entry = by_name.get(name) or by_source.get(name)
        if entry:
            entries.append(entry)
    entries.append(
        {
            "checkpoint_name": "gd_loc_s1000",
            "source_checkpoint_name": "gd_loc_s1000",
            "method_family": "gd",
            "checkpoint_path": "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors",
            "checkpoint_exists": Path("data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors").exists(),
            "source_selection_role": "retain_pass_gd_reference_for_task5b_v2",
            "retain_safety_flag": "pass",
        }
    )
    seen = set()
    unique = []
    for entry in entries:
        key = entry.get("checkpoint_name")
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    out = Path(args.task5b_v2_out_dir) / "task5b_v2_checkpoint_manifest.json"
    write_json(
        out,
        {
            "created_at": now(),
            "task": "task5b_v2_clean_probe_checkpoint_manifest",
            "source_task5a_manifest": args.task5a_manifest,
            "checkpoints": unique,
        },
    )
    write_task5b_v2_manifest_metadata(args, out, unique)
    return out


def assert_validity_gate(args: argparse.Namespace) -> None:
    audit = read_json(Path(args.task7r_out_dir) / "probe_validity" / "probe_validity_audit.json")
    decision = audit.get("decision", {})
    action = decision.get("action", "missing")
    if action.startswith("pause") or action == "missing":
        write_status(args, "stopped_validity_gate", validity_decision=decision)
        raise RuntimeError(f"Task 7-R validity gate failed: {decision}")
    print(f"[queue7r] validity gate passed with action={action}", flush=True)


def assert_task7_gate(args: argparse.Namespace) -> None:
    calibration = read_json(Path(args.task7r_out_dir) / "identity_capability_calibration.json")
    decision = calibration.get("decision", {})
    status = decision.get("capability_probe_status")
    if status != "clean_formal_gate":
        write_status(args, "stopped_task7r_gate", task7_decision=decision)
        raise RuntimeError(f"Task 7-R did not produce a clean formal gate: {decision}")
    print("[queue7r] Task 7-R clean formal gate passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task7r-out-dir", default="data/phase2/audits/task7r_capability_probe_20260714")
    parser.add_argument("--task8-out-dir", default="data/phase2/audits/task8_identity_capability_20260714")
    parser.add_argument("--task5b-v2-out-dir", default="data/phase2/audits/task5b_v2_clean_probe_20260714")
    parser.add_argument("--queue-status", default="data/phase2/audits/task5b_v2_clean_probe_20260714/task7r8_5bv2_queue_status.json")
    parser.add_argument("--task5a-out-dir", default="data/phase2/audits/task5a_identity_reaudit_20260713")
    parser.add_argument("--task5a-manifest", default="data/phase2/audits/task5a_identity_reaudit_20260713/task5a_for_task7_checkpoint_manifest.json")
    parser.add_argument("--benchmark-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--primary-task", default="hvue_human_transmissibility_coronaviridae")
    parser.add_argument("--aux-task", default="hvue_human_host_tropism")
    parser.add_argument("--formal-split-column", default="similarity_split")
    parser.add_argument("--max-per-split-label", type=int, default=400)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--layers", default="0-15")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--checkpoint-format", default="auto")
    parser.add_argument("--probe-seeds", default="42,43,44")
    parser.add_argument("--fresh-c-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--stop-on-low-disk-gb", type=float, default=60.0)
    args = parser.parse_args()

    Path(args.task7r_out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.task8_out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.task5b_v2_out_dir).mkdir(parents=True, exist_ok=True)
    write_queue_metadata(args)
    write_status(args, "started")

    require_disk(args, "task7r-dataset")
    run_command(
        args,
        [
            sys.executable,
            "phase2/build_capability_probe_dataset.py",
            "--benchmark-manifest", args.benchmark_manifest,
            "--fallback-benchmark-manifest", args.benchmark_manifest,
            "--primary-task", args.primary_task,
            "--aux-task", args.aux_task,
            "--out-dir", args.task7r_out_dir,
            "--formal-split-column", args.formal_split_column,
            "--max-per-split-label", str(args.max_per_split_label),
            "--n-bootstrap", str(args.n_bootstrap),
        ],
        "task7r-dataset",
    )

    require_disk(args, "task7r-validity")
    run_command(
        args,
        [
            sys.executable,
            "phase2/probe_validity_audit.py",
            "--internal-target-config", str(Path(args.task7r_out_dir) / "task7r_internal_target_config.json"),
            "--out-dir", str(Path(args.task7r_out_dir) / "probe_validity"),
            "--seeds", args.probe_seeds,
            "--c-grid", "0.001,0.01,0.1,1.0,10.0",
            "--n-bootstrap", str(args.n_bootstrap),
        ],
        "task7r-validity",
    )
    assert_validity_gate(args)

    require_disk(args, "task7r-probe")
    run_command(
        args,
        [
            sys.executable,
            "phase2/eval_capability_probe.py",
            "--dataset-manifest", str(Path(args.task7r_out_dir) / "capability_dataset_manifest.csv"),
            "--dataset-audit", str(Path(args.task7r_out_dir) / "capability_dataset_audit.json"),
            "--split-column", args.formal_split_column,
            "--checkpoint-manifest", args.task5a_manifest,
            "--out-dir", args.task7r_out_dir,
            "--layers", args.layers,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--model-dir", args.model_dir,
            "--config-path", args.config_path,
            "--checkpoint-format", args.checkpoint_format,
            "--seeds", args.probe_seeds,
            "--c-grid", args.fresh_c_grid,
        ],
        "task7r-probe",
    )
    run_command(
        args,
        [
            sys.executable,
            "phase2/summarize_identity_capability_calibration.py",
            "--mode", "task7",
            "--out-dir", args.task7r_out_dir,
            "--metrics", str(Path(args.task7r_out_dir) / "capability_probe_metrics.csv"),
            "--dataset-audit", str(Path(args.task7r_out_dir) / "capability_dataset_audit.json"),
            "--task5a-summary", str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
        ],
        "task7r-summary",
    )
    assert_task7_gate(args)

    run_command(
        args,
        [
            sys.executable,
            "phase2/run_task8_identity_capability_calibration.py",
            "--task7-dir", args.task7r_out_dir,
            "--task5a-summary", str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
            "--out-dir", args.task8_out_dir,
        ],
        "task8-calibration",
    )

    require_disk(args, "task5b-v2-probe")
    manifest = build_task5b_v2_manifest(args)
    run_command(
        args,
        [
            sys.executable,
            "phase2/eval_capability_probe.py",
            "--dataset-manifest", str(Path(args.task7r_out_dir) / "capability_dataset_manifest.csv"),
            "--dataset-audit", str(Path(args.task7r_out_dir) / "capability_dataset_audit.json"),
            "--split-column", args.formal_split_column,
            "--checkpoint-manifest", str(manifest),
            "--out-dir", args.task5b_v2_out_dir,
            "--layers", args.layers,
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--model-dir", args.model_dir,
            "--config-path", args.config_path,
            "--checkpoint-format", args.checkpoint_format,
            "--seeds", args.probe_seeds,
            "--c-grid", args.fresh_c_grid,
        ],
        "task5b-v2-probe",
    )
    run_command(
        args,
        [
            sys.executable,
            "phase2/summarize_identity_capability_calibration.py",
            "--mode", "task5b",
            "--out-dir", args.task5b_v2_out_dir,
            "--metrics", str(Path(args.task5b_v2_out_dir) / "capability_probe_metrics.csv"),
            "--dataset-audit", str(Path(args.task7r_out_dir) / "capability_dataset_audit.json"),
            "--task5a-summary", str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
            "--task7-calibration", str(Path(args.task7r_out_dir) / "identity_capability_calibration.json"),
        ],
        "task5b-v2-summary",
    )
    write_status(args, "completed")
    print("[queue7r] completed Task 7-R -> Task 8 -> Task 5B-v2", flush=True)


if __name__ == "__main__":
    main()
