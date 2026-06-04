"""Run a long command and notify when it exits.

This wrapper catches cases the child process cannot report itself, such as a
SIGKILL/OOM-kill exit. Put `--` before the command you want to run.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.notify import notify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="long job")
    parser.add_argument(
        "--webhook",
        default=os.environ.get("FEISHU_WEBHOOK", ""),
        help="Optional Feishu incoming webhook URL. Defaults to FEISHU_WEBHOOK.",
    )
    parser.add_argument("--sound", action="store_true", help="Emit a terminal bell on exit.")
    parser.add_argument(
        "--notify-success",
        action="store_true",
        help="Also notify on exit code 0.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Optional working directory for the child command.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def exit_reason(returncode: int) -> str:
    if returncode == 0:
        return "completed successfully"
    if returncode < 0:
        signum = -returncode
        try:
            return f"terminated by {signal.Signals(signum).name}"
        except ValueError:
            return f"terminated by signal {signum}"
    if returncode >= 128:
        signum = returncode - 128
        try:
            return f"terminated by {signal.Signals(signum).name}"
        except ValueError:
            return f"exited with code {returncode}"
    return f"exited with code {returncode}"


def run_child(command: List[str], cwd: Optional[str]) -> int:
    proc = subprocess.Popen(command, cwd=cwd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        return proc.wait()


def main() -> None:
    args = parse_args()
    started = time.time()
    returncode = run_child(args.command, args.cwd)
    elapsed_min = (time.time() - started) / 60.0
    reason = exit_reason(returncode)
    should_notify = returncode != 0 or args.notify_success
    if should_notify:
        title = f"[runner] {args.name}: {reason}"
        body = (
            f"name: {args.name}\n"
            f"returncode: {returncode}\n"
            f"elapsed min: {elapsed_min:.1f}\n"
            f"cwd: {args.cwd or os.getcwd()}\n"
            f"command: {' '.join(args.command)}"
        )
        notify(
            title=title,
            body=body,
            webhook_url=args.webhook or None,
            sound=args.sound,
        )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
