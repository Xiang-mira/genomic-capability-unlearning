"""Durable supervisor for the Stage 1 base-calibration queue.

This launcher is intentionally orchestration-only. It does not change the
training semantics inside eval_benchmarks.py or the current experiment plan.
Instead it:
1. starts the formal queue in a new session detached from the current terminal;
2. records pid/log/status metadata under the experiment out root;
3. auto-restarts the queue with --resume if it exits before all planned runs
   reach status=complete.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/stage1_formal_experiment_20260727"
DEFAULT_PLAN_JSON = DEFAULT_OUT_ROOT / "stage1_formal_experiment_plan.json"
DEFAULT_STATUS_JSON = DEFAULT_OUT_ROOT / "queue_supervisor_status.json"
DEFAULT_PID_FILE = DEFAULT_OUT_ROOT / "queue_supervisor.pid"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs/stage1_formal_queue_supervisor.log"
DEFAULT_HEARTBEAT_SEC = 30


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(json.loads(path.read_text()))


def collect_progress_counts(out_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in out_root.glob("fresh_lora/*/rank_*/lr_*/seed_*/eval_benchmarks_progress.json"):
        status = json.loads(path.read_text()).get("status", "missing")
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def all_runs_complete(out_root: Path, expected: int) -> bool:
    if expected <= 0:
        return False
    counts = collect_progress_counts(out_root)
    return counts.get("complete", 0) >= expected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def supervisor_status(
    *,
    state: str,
    attempt: int,
    pid: int | None,
    out_root: Path,
    plan_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": state,
        "attempt": attempt,
        "timestamp_utc": now_utc(),
        "pid": pid,
        "plan_count": plan_count,
        "progress_counts": collect_progress_counts(out_root),
    }
    if extra:
        payload.update(extra)
    return payload


def run_supervisor(args: argparse.Namespace) -> int:
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(f"{os.getpid()}\n")
    stop_requested = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        write_json(
            args.status_json,
            supervisor_status(
                state="stopping",
                attempt=attempt,
                pid=os.getpid(),
                out_root=args.out_root,
                plan_count=load_plan_count(args.plan_json),
                extra={"signal": signal.Signals(signum).name},
            ),
        )

    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled, handle_signal)

    attempt = 0
    while not stop_requested:
        plan_count = load_plan_count(args.plan_json)
        if all_runs_complete(args.out_root, plan_count):
            write_json(
                args.status_json,
                supervisor_status(
                    state="complete",
                    attempt=attempt,
                    pid=os.getpid(),
                    out_root=args.out_root,
                    plan_count=plan_count,
                ),
            )
            return 0

        attempt += 1
        cmd = [
            args.python_bin,
            "-u",
            "phase2/run_stage1_formal_experiment.py",
            "--checkpoint-mode",
            args.checkpoint_mode,
            "--execute",
        ]
        cmd.extend(args.runner_arg)
        if args.include_probe_vs_sft:
            cmd.append("--include-probe-vs-sft")
        with args.log_file.open("a") as log_handle:
            log_handle.write(f"\n[{now_utc()}] supervisor attempt={attempt} cmd={' '.join(cmd)}\n")
            log_handle.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            child_extra = {"child_pid": proc.pid, "cmd": cmd}
            write_json(
                args.status_json,
                supervisor_status(
                    state="running",
                    attempt=attempt,
                    pid=os.getpid(),
                    out_root=args.out_root,
                    plan_count=plan_count,
                    extra=child_extra,
                ),
            )
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    break
                write_json(
                    args.status_json,
                    supervisor_status(
                        state="running",
                        attempt=attempt,
                        pid=os.getpid(),
                        out_root=args.out_root,
                        plan_count=load_plan_count(args.plan_json),
                        extra=child_extra,
                    ),
                )
                time.sleep(args.heartbeat_sec)
            log_handle.write(f"[{now_utc()}] supervisor attempt={attempt} child_exit={returncode}\n")
            log_handle.flush()

        if stop_requested:
            break
        if all_runs_complete(args.out_root, plan_count):
            write_json(
                args.status_json,
                supervisor_status(
                    state="complete",
                    attempt=attempt,
                    pid=os.getpid(),
                    out_root=args.out_root,
                    plan_count=plan_count,
                    extra={"last_child_returncode": returncode},
                ),
            )
            return 0

        write_json(
            args.status_json,
            supervisor_status(
                state="restarting",
                attempt=attempt,
                pid=os.getpid(),
                out_root=args.out_root,
                plan_count=plan_count,
                extra={"last_child_returncode": returncode, "sleep_sec": args.restart_delay_sec},
            ),
        )
        time.sleep(args.restart_delay_sec)

    write_json(
        args.status_json,
        supervisor_status(
            state="stopped",
            attempt=attempt,
            pid=os.getpid(),
            out_root=args.out_root,
            plan_count=plan_count,
        ),
    )
    return 0


def launch_supervisor(args: argparse.Namespace) -> int:
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        "-u",
        "phase2/stage1_queue_supervisor.py",
        "run",
        "--python-bin",
        args.python_bin,
        "--checkpoint-mode",
        args.checkpoint_mode,
        "--out-root",
        str(args.out_root),
        "--plan-json",
        str(args.plan_json),
        "--status-json",
        str(args.status_json),
        "--pid-file",
        str(args.pid_file),
        "--log-file",
        str(args.log_file),
        "--restart-delay-sec",
        str(args.restart_delay_sec),
    ]
    for item in args.runner_arg:
        cmd.append(f"--runner-arg={item}")
    if args.include_probe_vs_sft:
        cmd.append("--include-probe-vs-sft")
    with args.log_file.open("a") as log_handle:
        log_handle.write(f"\n[{now_utc()}] launcher cmd={' '.join(cmd)}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(proc.pid)
    return 0


def stop_supervisor(args: argparse.Namespace) -> int:
    if not args.pid_file.exists():
        raise SystemExit(f"pid file not found: {args.pid_file}")
    pid = int(args.pid_file.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    return 0


def status_supervisor(args: argparse.Namespace) -> int:
    if not args.status_json.exists():
        raise SystemExit(f"status file not found: {args.status_json}")
    print(args.status_json.read_text(), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--python-bin", default=sys.executable)
        p.add_argument("--checkpoint-mode", default="base_only", choices=["base_only", "all", "modified_only"])
        p.add_argument("--include-probe-vs-sft", action="store_true")
        p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
        p.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
        p.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
        p.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE)
        p.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
        p.add_argument("--restart-delay-sec", type=int, default=15)
        p.add_argument("--heartbeat-sec", type=int, default=DEFAULT_HEARTBEAT_SEC)
        p.add_argument("--runner-arg", action="append", default=[], help="Extra argument passed through to run_stage1_formal_experiment.py")

    launch = sub.add_parser("launch")
    add_common(launch)

    run = sub.add_parser("run")
    add_common(run)

    stop = sub.add_parser("stop")
    add_common(stop)

    status = sub.add_parser("status")
    add_common(status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "launch":
        raise SystemExit(launch_supervisor(args))
    if args.command == "run":
        raise SystemExit(run_supervisor(args))
    if args.command == "stop":
        raise SystemExit(stop_supervisor(args))
    if args.command == "status":
        raise SystemExit(status_supervisor(args))
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
