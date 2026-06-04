"""Watch a long eval run and notify if it exits without a terminal progress state."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.notify import notify


TERMINAL_STATUSES = {"complete", "failed", "interrupted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--progress", action="append", default=[])
    parser.add_argument("--log", default="")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--grace-sec", type=float, default=10.0)
    parser.add_argument(
        "--webhook",
        default=os.environ.get("FEISHU_WEBHOOK", ""),
        help="Optional Feishu incoming webhook URL. Defaults to FEISHU_WEBHOOK.",
    )
    parser.add_argument("--sound", action="store_true")
    parser.add_argument("--notify-success", action="store_true")
    return parser.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def load_progress(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        payload["_path"] = path
        payload["_mtime"] = os.path.getmtime(path)
        return payload
    except (OSError, json.JSONDecodeError):
        return None


def progress_snapshot(paths: List[str]) -> List[Dict[str, object]]:
    payloads = [payload for path in paths if (payload := load_progress(path)) is not None]
    return sorted(payloads, key=lambda item: float(item.get("_mtime", 0.0)), reverse=True)


def all_complete(payloads: List[Dict[str, object]], expected_count: int) -> bool:
    if len(payloads) < expected_count:
        return False
    return all(str(payload.get("status", "")) == "complete" for payload in payloads)


def format_progress(payloads: List[Dict[str, object]]) -> str:
    if not payloads:
        return "progress: none"
    lines = []
    for payload in payloads:
        lines.append(
            "progress: "
            f"{payload.get('_path')} "
            f"status={payload.get('status')} "
            f"task={payload.get('task_index')}/{payload.get('task_total')} "
            f"phase={payload.get('phase')} "
            f"current_task={payload.get('current_task', '')} "
            f"completed={payload.get('completed_task_layers')}/{payload.get('expected_task_layers')}"
        )
    return "\n".join(lines)


def send(args: argparse.Namespace, title: str, detail: str) -> None:
    body = (
        f"name: {args.name}\n"
        f"pid: {args.pid}\n"
        f"log: {args.log}\n"
        f"{detail}"
    )
    notify(title=title, body=body, webhook_url=args.webhook or None, sound=args.sound)


def main() -> None:
    args = parse_args()
    while True:
        payloads = progress_snapshot(args.progress)
        statuses = {str(payload.get("status", "")) for payload in payloads}

        if all_complete(payloads, len(args.progress)):
            if args.notify_success:
                send(args, f"[watchdog] {args.name}: complete", format_progress(payloads))
            return

        if statuses & {"failed", "interrupted"}:
            send(args, f"[watchdog] {args.name}: {', '.join(sorted(statuses & TERMINAL_STATUSES))}", format_progress(payloads))
            return

        if not pid_alive(args.pid):
            time.sleep(max(0.0, args.grace_sec))
            payloads = progress_snapshot(args.progress)
            statuses = {str(payload.get("status", "")) for payload in payloads}
            if all_complete(payloads, len(args.progress)):
                if args.notify_success:
                    send(args, f"[watchdog] {args.name}: complete", format_progress(payloads))
                return
            send(
                args,
                f"[watchdog] {args.name}: process exited without terminal progress",
                format_progress(payloads),
            )
            return

        time.sleep(max(1.0, args.poll_sec))


if __name__ == "__main__":
    main()
