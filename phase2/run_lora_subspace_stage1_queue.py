"""Run pending Stage 1 LoRA-subspace targeted reruns serially.

This is intentionally a thin queue wrapper around the formal rerun registry.
It does not start Stage 2. After each rerun it refreshes the registry so a
later analysis step can decide the Stage 1 Go/No-Go from current artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from phase2.lora_subspace_targeting import build_parser as build_scaffold_parser
from phase2.lora_subspace_targeting import run as refresh_scaffold


DEFAULT_OUT_ROOT = Path("data/phase2/lora_subspace_targeting_20260729")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_plan(out_dir: Path) -> dict[str, object]:
    return json.loads((out_dir / "missing_artifacts_rerun_plan.json").read_text())


def refresh(out_dir: Path) -> None:
    parser = build_scaffold_parser()
    args = parser.parse_args(["--out-dir", str(out_dir)])
    refresh_scaffold(args)


def run_queue(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    log_path = args.log_path or out_dir / "stage1_rerun_queue.log"
    status_path = out_dir / "stage1_rerun_queue_status.json"
    refresh(out_dir)
    plan = load_plan(out_dir)
    planned = [
        row
        for row in plan["planned_reruns"]
        if row.get("command") and row.get("status") != "complete"
    ]
    if args.max_runs > 0:
        planned = planned[: args.max_runs]
    state = {
        "status": "running",
        "started_at_utc": utc_now(),
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "planned_run_count": len(planned),
        "completed": [],
        "failed": [],
    }
    write_json(status_path, state)
    with log_path.open("a") as log:
        log.write(f"[{utc_now()}] queue_start planned={len(planned)}\n")
        log.flush()
        for row in planned:
            run_id = str(row["run_id"])
            command = [str(part) for part in row["command"]]
            state["current_run_id"] = run_id
            state["current_command"] = command
            state["updated_at_utc"] = utc_now()
            write_json(status_path, state)
            log.write(f"[{utc_now()}] run_start {run_id}\n")
            log.write("COMMAND " + " ".join(command) + "\n")
            log.flush()
            started = time.time()
            result = subprocess.run(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - started
            refresh(out_dir)
            if result.returncode == 0:
                state["completed"].append({"run_id": run_id, "elapsed_sec": elapsed})
                log.write(f"[{utc_now()}] run_complete {run_id} elapsed_sec={elapsed:.3f}\n")
            else:
                state["failed"].append({"run_id": run_id, "elapsed_sec": elapsed, "returncode": result.returncode})
                state["status"] = "failed"
                state["updated_at_utc"] = utc_now()
                write_json(status_path, state)
                log.write(f"[{utc_now()}] run_failed {run_id} returncode={result.returncode} elapsed_sec={elapsed:.3f}\n")
                log.flush()
                if not args.keep_going:
                    return
            log.flush()
            state["updated_at_utc"] = utc_now()
            write_json(status_path, state)
    refresh(out_dir)
    state["status"] = "complete" if not state["failed"] else "complete_with_failures"
    state["completed_at_utc"] = utc_now()
    state.pop("current_run_id", None)
    state.pop("current_command", None)
    write_json(status_path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main() -> None:
    run_queue(build_parser().parse_args())


if __name__ == "__main__":
    main()
