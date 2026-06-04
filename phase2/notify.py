"""Small notification helpers for long-running phase2 jobs."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Optional


def terminal_bell() -> None:
    """Emit a terminal bell. It is best-effort and harmless in non-TTY logs."""
    try:
        sys.stderr.write("\a")
        sys.stderr.flush()
    except Exception:
        pass


def send_feishu_text(webhook_url: str, text: str, timeout: float = 10.0) -> bool:
    """Send a plain text message to a Feishu incoming webhook."""
    payload = {
        "msg_type": "text",
        "content": {"text": text},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
        return True
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"[notify] failed to send Feishu webhook: {exc}", file=sys.stderr)
        return False


def notify(
    *,
    title: str,
    body: str,
    webhook_url: Optional[str] = None,
    sound: bool = False,
) -> None:
    if sound:
        terminal_bell()
    if webhook_url:
        send_feishu_text(webhook_url, f"{title}\n{body}")
