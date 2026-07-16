"""Automatic screen-friendly queue for Task 5A, Task 7, and Task 5B.

The queue is conservative by design: it resumes completed artifacts, records
missing checkpoints, retries per-checkpoint failures once, checks disk before
heavy runs, and continues past single-checkpoint failures.
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

from phase2.run_task5a_identity_reaudit import TASK3_CONTEXT, TASK5A_CHECKPOINTS


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


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


class QueueStopped(RuntimeError):
    pass


def write_queue_status(args: argparse.Namespace, status: str, **extra: Any) -> None:
    payload = {
        "updated_at": now(),
        "status": status,
        "task3_context": TASK3_CONTEXT,
        "task5a_out_dir": args.task5a_out_dir,
        "task7_out_dir": args.task7_out_dir,
        "task5b_out_dir": args.task5b_out_dir,
        **extra,
    }
    write_json(Path(args.task5b_out_dir) / "task5ab7_queue_status.json", payload)


def require_disk(args: argparse.Namespace, path: Path, stage: str) -> None:
    free = free_disk_gb(path)
    if free < args.stop_on_low_disk_gb:
        write_queue_status(
            args,
            "stopped_low_disk",
            stage=stage,
            free_disk_gb=free,
            stop_on_low_disk_gb=args.stop_on_low_disk_gb,
        )
        raise QueueStopped(
            f"low disk before {stage}: free={free:.2f}G threshold={args.stop_on_low_disk_gb:.2f}G"
        )
    print(f"[queue] disk ok before {stage}: free={free:.2f}G threshold={args.stop_on_low_disk_gb:.2f}G", flush=True)


def run_command(command: list[str], env: dict[str, str], stage: str) -> int:
    print(f"[queue] start {stage}: {' '.join(command)}", flush=True)
    started = time.time()
    result = subprocess.run(command, env=env)
    elapsed = time.time() - started
    print(f"[queue] finish {stage}: returncode={result.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    return int(result.returncode)


def task5a_completed(out_root: Path, checkpoint_name: str) -> bool:
    out_dir = out_root / checkpoint_name
    status = read_json(out_dir / "status.json")
    required = ["eval_auroc.csv", "eval_ppl.json", "eval_representation.csv", "meta.json"]
    return status.get("run_status") == "completed" and all((out_dir / name).exists() for name in required)


def run_task5a_checkpoint(args: argparse.Namespace, env: dict[str, str], checkpoint_name: str) -> int:
    out_root = Path(args.task5a_out_dir)
    if args.resume and task5a_completed(out_root, checkpoint_name):
        print(f"[queue] Task 5A skip completed {checkpoint_name}", flush=True)
        return 0

    final_code = 0
    for attempt in range(args.retry_on_failure + 1):
        batch_size = args.batch_size if attempt == 0 else args.oom_retry_batch_size
        require_disk(args, out_root, f"task5a:{checkpoint_name}:attempt{attempt + 1}")
        command = [
            sys.executable,
            "phase2/run_task5a_identity_reaudit.py",
            "--out-root",
            args.task5a_out_dir,
            "--checkpoint-name",
            checkpoint_name,
            "--batch-size",
            str(batch_size),
            "--device",
            args.device,
            "--model-dir",
            args.model_dir,
            "--config-path",
            args.config_path,
            "--checkpoint-format",
            args.checkpoint_format,
        ]
        if args.resume:
            command.append("--resume")
        code = run_command(command, env, f"task5a:{checkpoint_name}:batch{batch_size}")
        if code == 0:
            return 0
        final_code = code
        print(
            f"[queue] Task 5A checkpoint failed {checkpoint_name} attempt={attempt + 1} code={code}",
            flush=True,
        )
    print(f"[queue] Task 5A giving up on {checkpoint_name}; continuing queue", flush=True)
    return final_code


def summarize_task5a(args: argparse.Namespace, env: dict[str, str]) -> int:
    command = [
        sys.executable,
        "phase2/summarize_task5a_identity_reaudit.py",
        "--out-root",
        args.task5a_out_dir,
    ]
    return run_command(command, env, "task5a-summary")


def wait_for_task7_ready(args: argparse.Namespace) -> None:
    if not args.wait_for_task7_ready:
        return
    flag = Path(args.task7_ready_flag)
    while not flag.exists():
        print(
            f"[queue] waiting for Task 7 ready flag: {flag}; sleeping {args.ready_poll_seconds}s",
            flush=True,
        )
        write_queue_status(args, "waiting_for_task7_ready", task7_ready_flag=str(flag))
        time.sleep(args.ready_poll_seconds)
    print(f"[queue] Task 7 ready flag found: {flag}", flush=True)


def task7_dataset_ready(out_dir: Path) -> bool:
    return (out_dir / "capability_dataset_manifest.csv").exists() and (out_dir / "capability_dataset_audit.json").exists()


def task7_eval_ready(out_dir: Path) -> bool:
    return (out_dir / "capability_probe_metrics.csv").exists() and (out_dir / "identity_capability_calibration.json").exists()


def task5b_ready(out_dir: Path) -> bool:
    required = [
        "capability_probe_metrics.csv",
        "task5b_capability_reaudit_summary.csv",
        "task5b_capability_reaudit_summary.json",
        "task5b_decision.md",
        "task5ab7_joint_decision.md",
        "p5_initialization_candidates.json",
    ]
    return all((out_dir / name).exists() for name in required)


def run_task7_dataset(args: argparse.Namespace, env: dict[str, str]) -> int:
    out_dir = Path(args.task7_out_dir)
    if args.resume and task7_dataset_ready(out_dir):
        print("[queue] Task 7 dataset skip completed artifacts", flush=True)
        return 0
    require_disk(args, out_dir, "task7-dataset")
    command = [
        sys.executable,
        "phase2/build_capability_probe_dataset.py",
        "--out-dir",
        args.task7_out_dir,
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--max-per-split-label",
        str(args.max_eval),
    ]
    return run_command(command, env, "task7-dataset")


def task7_manifest_path(args: argparse.Namespace) -> Path:
    return Path(args.task5a_out_dir) / "task5a_for_task7_checkpoint_manifest.json"


def run_task7_probe(args: argparse.Namespace, env: dict[str, str]) -> int:
    out_dir = Path(args.task7_out_dir)
    if args.resume and task7_eval_ready(out_dir):
        print("[queue] Task 7 probe skip completed artifacts", flush=True)
        return 0
    require_disk(args, out_dir, "task7-probe")
    command = [
        sys.executable,
        "phase2/eval_capability_probe.py",
        "--dataset-manifest",
        str(out_dir / "capability_dataset_manifest.csv"),
        "--dataset-audit",
        str(out_dir / "capability_dataset_audit.json"),
        "--checkpoint-manifest",
        str(task7_manifest_path(args)),
        "--out-dir",
        args.task7_out_dir,
        "--layers",
        args.layers,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--model-dir",
        args.model_dir,
        "--config-path",
        args.config_path,
        "--checkpoint-format",
        args.checkpoint_format,
        "--seeds",
        args.probe_seeds,
        "--c-grid",
        args.fresh_c_grid,
    ]
    code = run_command(command, env, "task7-probe")
    if code != 0:
        return code
    command = [
        sys.executable,
        "phase2/summarize_identity_capability_calibration.py",
        "--mode",
        "task7",
        "--out-dir",
        args.task7_out_dir,
        "--metrics",
        str(out_dir / "capability_probe_metrics.csv"),
        "--dataset-audit",
        str(out_dir / "capability_dataset_audit.json"),
        "--task5a-summary",
        str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
    ]
    return run_command(command, env, "task7-summary")


def build_task5b_manifest(args: argparse.Namespace) -> Path:
    task5a_manifest = read_json(task7_manifest_path(args))
    entries = list(task5a_manifest.get("checkpoints", []))
    seen_names = {entry.get("checkpoint_name") for entry in entries}
    seen_sources = {entry.get("source_checkpoint_name", entry.get("checkpoint_name")) for entry in entries}
    task5a_summary = read_json(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json")
    rows = task5a_summary.get("rows", [])
    optional_controls = {"projection_rank16", "gd_random_control", "rmu_joint_sc50_ar5"}
    for row in rows:
        name = row.get("checkpoint_name")
        if name not in optional_controls:
            continue
        if row.get("run_status") != "completed":
            continue
        if not row.get("recommended_for_capability_reaudit") and not row.get("recommended_for_p5_init"):
            continue
        if name in seen_names or name in seen_sources:
            continue
        entries.append(
            {
                "checkpoint_name": name,
                "source_checkpoint_name": name,
                "method_family": row.get("method_family", "unknown"),
                "checkpoint_path": row.get("checkpoint_path", ""),
                "checkpoint_exists": row.get("checkpoint_exists", True),
                "source_selection_role": "task5b_optional_control_if_informative",
                "task5a_run_status": row.get("run_status"),
                "retain_safety_flag": row.get("retain_safety_flag"),
                "fresh_family_mean_separability": row.get("fresh_family_mean_separability"),
                "fresh_family_max_separability": row.get("fresh_family_max_separability"),
            }
        )
        seen_names.add(name)
        seen_sources.add(name)

    out_path = Path(args.task5b_out_dir) / "task5b_checkpoint_manifest.json"
    write_json(
        out_path,
        {
            "created_at": now(),
            "task": "task5b_capability_reaudit_checkpoint_manifest",
            "source_task5a_manifest": str(task7_manifest_path(args)),
            "task3_context": TASK3_CONTEXT,
            "checkpoints": entries,
        },
    )
    return out_path


def run_task5b_probe(args: argparse.Namespace, env: dict[str, str]) -> int:
    out_dir = Path(args.task5b_out_dir)
    if args.resume and task5b_ready(out_dir):
        print("[queue] Task 5B skip completed artifacts", flush=True)
        return 0
    require_disk(args, out_dir, "task5b-probe")
    manifest = build_task5b_manifest(args)
    command = [
        sys.executable,
        "phase2/eval_capability_probe.py",
        "--dataset-manifest",
        str(Path(args.task7_out_dir) / "capability_dataset_manifest.csv"),
        "--dataset-audit",
        str(Path(args.task7_out_dir) / "capability_dataset_audit.json"),
        "--checkpoint-manifest",
        str(manifest),
        "--out-dir",
        args.task5b_out_dir,
        "--layers",
        args.layers,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--model-dir",
        args.model_dir,
        "--config-path",
        args.config_path,
        "--checkpoint-format",
        args.checkpoint_format,
        "--seeds",
        args.probe_seeds,
        "--c-grid",
        args.fresh_c_grid,
    ]
    code = run_command(command, env, "task5b-probe")
    if code != 0:
        return code
    command = [
        sys.executable,
        "phase2/summarize_identity_capability_calibration.py",
        "--mode",
        "task5b",
        "--out-dir",
        args.task5b_out_dir,
        "--metrics",
        str(out_dir / "capability_probe_metrics.csv"),
        "--dataset-audit",
        str(Path(args.task7_out_dir) / "capability_dataset_audit.json"),
        "--task5a-summary",
        str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
        "--task7-calibration",
        str(Path(args.task7_out_dir) / "identity_capability_calibration.json"),
    ]
    return run_command(command, env, "task5b-summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-low-disk-gb", type=float, default=80.0)
    parser.add_argument("--task5a-out-dir", default="data/phase2/audits/task5a_identity_reaudit_20260713")
    parser.add_argument("--task7-out-dir", default="data/phase2/audits/task7_capability_probe_20260713")
    parser.add_argument("--task5b-out-dir", default="data/phase2/audits/task5b_capability_reaudit_20260713")
    parser.add_argument("--wait-for-task7-ready", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--task7-ready-flag",
        default="data/phase2/audits/task7_capability_probe_20260713/task7_code_ready.flag",
    )
    parser.add_argument("--ready-poll-seconds", type=int, default=300)
    parser.add_argument("--retry-on-failure", type=int, default=1)
    parser.add_argument("--oom-retry-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval", type=int, default=400)
    parser.add_argument("--layers", default="0-15")
    parser.add_argument("--probe-seeds", default="42,43,44")
    parser.add_argument("--fresh-c-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--checkpoint-format", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.task5a_out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.task7_out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.task5b_out_dir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    write_queue_status(args, "running", started_at=now())

    try:
        print("[queue] Task 5A starting", flush=True)
        failed = []
        for spec in TASK5A_CHECKPOINTS:
            code = run_task5a_checkpoint(args, env, spec.checkpoint_name)
            if code != 0:
                failed.append({"checkpoint_name": spec.checkpoint_name, "returncode": code})
        summary_code = summarize_task5a(args, env)
        if summary_code != 0:
            write_queue_status(args, "failed_task5a_summary", returncode=summary_code, failed_checkpoints=failed)
            raise SystemExit(summary_code)

        wait_for_task7_ready(args)
        dataset_code = run_task7_dataset(args, env)
        if dataset_code != 0:
            write_queue_status(args, "failed_task7_dataset", returncode=dataset_code, failed_checkpoints=failed)
            raise SystemExit(dataset_code)
        task7_code = run_task7_probe(args, env)
        if task7_code != 0:
            write_queue_status(args, "failed_task7_probe", returncode=task7_code, failed_checkpoints=failed)
            raise SystemExit(task7_code)
        task5b_code = run_task5b_probe(args, env)
        if task5b_code != 0:
            write_queue_status(args, "failed_task5b_probe", returncode=task5b_code, failed_checkpoints=failed)
            raise SystemExit(task5b_code)

        write_queue_status(args, "completed", completed_at=now(), failed_checkpoints=failed)
        print("[queue] completed Task 5A + Task 7 + Task 5B queue", flush=True)
    except QueueStopped as exc:
        print(f"[queue] stopped: {exc}", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
