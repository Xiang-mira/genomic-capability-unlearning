"""Preflight checks for the route-decision pipeline.

This script freezes the execution environment before any long-running screen
session starts. It writes machine-readable artifacts and exits non-zero on the
first hard failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.run_metadata import build_run_metadata, file_sha256, runtime_environment, write_metadata


PROJECT_ROOT = Path("/home/teacher1/UT-project1/project1")
DEFAULT_PYTHON = "/home/teacher1/miniconda3/envs/UT-p1/bin/python"
DEFAULT_MANIFEST = "data/benchmarks/hvue_gue_pilot_slim_manifest.csv"
DEFAULT_BASE_SUMMARY = "data/phase2/base_benchmarks_slim/eval_benchmarks_summary.json"
DEFAULT_RETAIN_CSV = "data/phase2/splits/retain.csv"


CHECKPOINTS = {
    "base": None,
    "projection_rank32": "data/phase2/checkpoints_projection_adaptive_rank32/projopt_host5_9_coro0_10_adaptive_basis_rank32/weights.safetensors",
    "best_gd_from_task5a": "data/phase2/checkpoints_tuned/refseq_gd_projinit_full_ar5_s200/weights.safetensors",
    "gd_random_control": "data/phase2/checkpoints_tuned/refseq_gd_projinit_random_ar5_s1000/weights.safetensors",
    "gd_loc_s1000": "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors",
}


REQUIRED_SCRIPTS = [
    "phase2/preflight_route_decision.py",
    "phase2/freeze_workspace_state.py",
    "phase2/run_route_decision_pipeline.py",
    "phase2/summarize_route_decision.py",
    "phase2/eval_benchmarks.py",
    "phase2/verify_retain_set.py",
]


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_manifest_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row["task"])
    return sorted(set(tasks))


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def resolve_workspace_snapshot_dir(
    project_root: Path,
    preflight_dir: Path,
    requested: str,
) -> Path:
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return (preflight_dir / "workspace_state").resolve()


def workspace_snapshot_command(
    *,
    args: argparse.Namespace,
    project_root: Path,
    preflight_dir: Path,
) -> tuple[list[str], Path]:
    snapshot_dir = resolve_workspace_snapshot_dir(project_root, preflight_dir, args.workspace_snapshot_out_dir)
    command = [
        args.python_bin,
        "-u",
        "phase2/freeze_workspace_state.py",
        "--project-root",
        str(project_root),
        "--out-dir",
        relative_to_project(snapshot_dir, project_root),
    ]
    return command, snapshot_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--out-dir", default="data/phase2/route_decision_20260715")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--retain-csv", default=DEFAULT_RETAIN_CSV)
    parser.add_argument("--base-summary", default=DEFAULT_BASE_SUMMARY)
    parser.add_argument("--task5a-summary", default="data/phase2/audits/task5a_identity_reaudit_20260713/task5a_identity_reaudit_summary.json")
    parser.add_argument("--task5b-summary", default="data/phase2/audits/task5b_capability_reaudit_20260713/task5b_capability_reaudit_summary.csv")
    parser.add_argument("--task7r-dir", default="data/phase2/audits/task7r_capability_probe_20260714")
    parser.add_argument("--task7s-dir", default="data/phase2/audits/task7s_clean_gate_20260715")
    parser.add_argument("--workspace-snapshot-out-dir", default="")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--validation-max-rows", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--metric-for-best", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    preflight_dir = out_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    env_report = runtime_environment()
    env_report["expected_python"] = args.python_bin
    env_report["requested_device"] = args.device
    env_report["project_root"] = str(project_root)
    env_report["cwd_matches_project_root"] = env_report["cwd"] == str(project_root)
    env_report["python_executable"] = sys.executable
    env_report["which_python"] = shutil.which("python")
    env_report["device_available"] = bool(torch.cuda.is_available()) if args.device.startswith("cuda") else True
    write_json(preflight_dir / "preflight_env.json", env_report)

    assert_ok(project_root == PROJECT_ROOT, f"project_root mismatch: {project_root} != {PROJECT_ROOT}")
    assert_ok(env_report["cwd_matches_project_root"], f"cwd must be {project_root}, got {env_report['cwd']}")
    assert_ok(Path(args.python_bin).exists(), f"python not found: {args.python_bin}")
    assert_ok(Path(sys.executable).resolve() == Path(args.python_bin).resolve(), f"sys.executable={sys.executable} != {args.python_bin}")
    assert_ok(env_report["which_python"] is not None, "python not on PATH")
    assert_ok(Path(env_report["which_python"]).resolve() == Path(args.python_bin).resolve(), f"which python={env_report['which_python']} != {args.python_bin}")
    if args.device.startswith("cuda"):
        assert_ok(torch.cuda.is_available(), "CUDA device requested but torch.cuda.is_available() is false")

    paths_report: dict[str, Any] = {
        "updated_at": now(),
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "required_scripts": {},
        "required_inputs": {},
        "checkpoints": {},
    }
    for rel in REQUIRED_SCRIPTS:
        path = project_root / rel
        paths_report["required_scripts"][rel] = {"exists": path.exists(), "sha256": file_sha256(path) if path.exists() else "missing"}
        assert_ok(path.exists(), f"missing script: {path}")

    required_inputs = {
        "benchmark_manifest": args.benchmark_manifest,
        "retain_csv": args.retain_csv,
        "base_summary": args.base_summary,
        "task5a_summary": args.task5a_summary,
        "task5b_summary": args.task5b_summary,
        "task7r_dir": args.task7r_dir,
        "task7s_dir": args.task7s_dir,
    }
    for name, rel in required_inputs.items():
        path = (project_root / rel).resolve()
        exists = path.exists()
        paths_report["required_inputs"][name] = {"path": str(path), "exists": exists}
        assert_ok(exists, f"missing input {name}: {path}")

    for name, rel in CHECKPOINTS.items():
        if rel is None:
            paths_report["checkpoints"][name] = {"path": None, "exists": True}
            continue
        path = (project_root / rel).resolve()
        exists = path.exists()
        paths_report["checkpoints"][name] = {"path": str(path), "exists": exists}
        assert_ok(exists, f"missing checkpoint {name}: {path}")

    write_json(preflight_dir / "preflight_paths.json", paths_report)

    benchmark_manifest = (project_root / args.benchmark_manifest).resolve()
    manifest_tasks = load_manifest_tasks(benchmark_manifest)
    assert_ok(
        manifest_tasks
        == [
            "gue_emp_h3",
            "gue_human_tf_1",
            "gue_mouse_1",
            "gue_prom_300_notata",
            "gue_splice_reconstructed",
            "hvue_human_host_tropism",
            "hvue_human_transmissibility_coronaviridae",
        ],
        f"unexpected slim manifest tasks: {manifest_tasks}",
    )

    verify_cmd = [
        args.python_bin,
        "-u",
        "phase2/verify_retain_set.py",
        "--csv",
        args.retain_csv,
        "--summary-json",
        str((preflight_dir / "retain_audit.json").relative_to(project_root)),
    ]
    verify_result = run_command(verify_cmd, project_root)
    assert_ok(verify_result.returncode == 0, f"verify_retain_set failed:\nSTDOUT:\n{verify_result.stdout}\nSTDERR:\n{verify_result.stderr}")

    for rel in REQUIRED_SCRIPTS:
        compile_result = run_command([args.python_bin, "-m", "py_compile", rel], project_root)
        assert_ok(compile_result.returncode == 0, f"py_compile failed for {rel}:\n{compile_result.stderr}")

    workspace_snapshot_cmd, workspace_snapshot_dir = workspace_snapshot_command(
        args=args,
        project_root=project_root,
        preflight_dir=preflight_dir,
    )
    workspace_snapshot = run_command(workspace_snapshot_cmd, project_root)
    assert_ok(
        workspace_snapshot.returncode == 0,
        f"freeze_workspace_state failed:\nSTDOUT:\n{workspace_snapshot.stdout}\nSTDERR:\n{workspace_snapshot.stderr}",
    )

    batch_preflight_cmd = [
        args.python_bin,
        "-u",
        "phase2/eval_benchmarks.py",
        "--preflight-only",
        "--device",
        args.device,
        "--cpu-threads",
        str(args.cpu_threads),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--validation-max-rows",
        str(args.validation_max_rows),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--metric-for-best",
        args.metric_for_best,
        "--seed",
        str(args.seed),
    ]
    batch_preflight = run_command(batch_preflight_cmd, project_root)
    write_json(
        preflight_dir / "preflight_inputs.json",
        {
            "updated_at": now(),
            "benchmark_manifest": {
                "path": str(benchmark_manifest),
                "sha256": file_sha256(benchmark_manifest),
                "tasks": manifest_tasks,
            },
            "workspace_snapshot": {
                "command": workspace_snapshot_cmd,
                "returncode": workspace_snapshot.returncode,
                "out_dir": str(workspace_snapshot_dir),
                "stdout_tail": workspace_snapshot.stdout.splitlines()[-20:],
                "stderr_tail": workspace_snapshot.stderr.splitlines()[-20:],
            },
            "batch_preflight": {
                "command": batch_preflight_cmd,
                "returncode": batch_preflight.returncode,
                "stdout_tail": batch_preflight.stdout.splitlines()[-20:],
                "stderr_tail": batch_preflight.stderr.splitlines()[-20:],
            },
        },
    )
    assert_ok(batch_preflight.returncode == 0, f"eval_benchmarks preflight failed:\nSTDOUT:\n{batch_preflight.stdout}\nSTDERR:\n{batch_preflight.stderr}")

    lock = {
        "created_at": now(),
        "objective": "route_decision_dual_axis",
        "python_bin": args.python_bin,
        "project_root": str(project_root),
        "device": args.device,
        "benchmark_manifest": str(benchmark_manifest),
        "base_summary": str((project_root / args.base_summary).resolve()),
        "retain_csv": str((project_root / args.retain_csv).resolve()),
        "task5a_summary": str((project_root / args.task5a_summary).resolve()),
        "task5b_summary": str((project_root / args.task5b_summary).resolve()),
        "task7r_dir": str((project_root / args.task7r_dir).resolve()),
        "task7s_dir": str((project_root / args.task7s_dir).resolve()),
        "workspace_state_snapshot_dir": str(workspace_snapshot_dir),
        "checkpoints": {
            name: (None if rel is None else str((project_root / rel).resolve())) for name, rel in CHECKPOINTS.items()
        },
        "execution_order": [
            "preflight",
            "freeze_current_state",
            "benchmark:gd_random_control",
            "benchmark:best_gd_from_task5a",
            "benchmark:projection_rank32",
            "reuse:base",
            "reuse:gd_loc_s1000",
            "summary",
            "go_no_go",
            "controlled_diagnostic_if_go",
            "final_report",
        ],
        "metrics": [
            "target_drop_mean",
            "retain_delta_mean",
            "retain_ppl",
            "random_gap",
            "absolute_signal_flag",
            "retain_flag",
            "decision",
            "evidence_note",
        ],
        "benchmark_settings": {
            "cpu_threads": args.cpu_threads,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "validation_max_rows": args.validation_max_rows,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "metric_for_best": args.metric_for_best,
            "seed": args.seed,
            "discard_task_checkpoint": True,
            "benchmark_scope": "all",
        },
    }
    write_json(preflight_dir / "run_manifest_lock.json", lock)
    write_metadata(
        preflight_dir / "preflight_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[args.benchmark_manifest, args.retain_csv, args.task5a_summary, args.task5b_summary, str(workspace_snapshot_dir)],
            extra={
                "phase": "route_decision_preflight",
                "workspace_state_snapshot_dir": str(workspace_snapshot_dir),
            },
        ),
    )
    print(f"[route-preflight] complete: {preflight_dir}", flush=True)


if __name__ == "__main__":
    main()
