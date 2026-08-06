"""Reproducibility metadata helpers for Phase 2 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing_paths(paths: Iterable[str | os.PathLike[str]]) -> dict[str, str]:
    result = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            result[str(path)] = file_sha256(path)
        elif path.exists() and path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            result[str(path)] = stable_hash([(str(item.relative_to(path)), file_sha256(item)) for item in files])
        else:
            result[str(path)] = "missing"
    return result


def git_info() -> dict[str, object]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run_git(["status", "--short"])
    return {
        "commit_hash": run_git(["rev-parse", "HEAD"]),
        "commit_subject": run_git(["log", "-1", "--pretty=%s"]),
        "git_dirty": bool(status),
        "git_status_short": status.splitlines() if status else [],
    }


def runtime_environment() -> dict[str, object]:
    try:
        import torch
    except Exception:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "torch": "unavailable",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        }

    cuda_available = torch.cuda.is_available()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    }


def namespace_to_dict(args: argparse.Namespace | Mapping[str, object] | None) -> dict[str, object]:
    if args is None:
        return {}
    if isinstance(args, Mapping):
        return dict(args)
    return vars(args).copy()


def build_run_metadata(
    *,
    args: argparse.Namespace | Mapping[str, object] | None = None,
    source_checkpoint: str | None = None,
    init_checkpoint: str | None = None,
    output_checkpoint: str | None = None,
    data_paths: Iterable[str] = (),
    probe_paths: Iterable[str] = (),
    basis_paths: Iterable[str] = (),
    trainable_modules: Iterable[str] = (),
    trainable_tensor_names: Iterable[str] = (),
    trainable_param_count: int = 0,
    loss_layers: Iterable[int] = (),
    hook_location: str = "next_norm",
    seed: int | None = None,
    save_policy: str | None = None,
    checkpoint_policy: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    args_dict = namespace_to_dict(args)
    git = git_info()
    basis_hashes = hash_existing_paths(basis_paths)
    metadata = {
        "source_checkpoint": source_checkpoint or args_dict.get("model_dir") or "base_model",
        "init_checkpoint": init_checkpoint or args_dict.get("init_ckpt") or "",
        "output_checkpoint": output_checkpoint or "",
        "config_hash": stable_hash(args_dict),
        "commit_hash": git["commit_hash"],
        "git_dirty": git["git_dirty"],
        "git_status_short": git["git_status_short"],
        "data_hashes": hash_existing_paths(data_paths),
        "probe_hashes": hash_existing_paths(probe_paths),
        "basis_checksum": stable_hash(basis_hashes) if basis_hashes else "none",
        "basis_hashes": basis_hashes,
        "trainable_modules": list(trainable_modules),
        "trainable_tensor_names": list(trainable_tensor_names),
        "trainable_param_count": int(trainable_param_count),
        "loss_layers": [int(layer) for layer in loss_layers],
        "hook_location": hook_location,
        "seed": seed if seed is not None else args_dict.get("seed"),
        "save_policy": save_policy or args_dict.get("save_policy", ""),
        "checkpoint_policy": checkpoint_policy or save_policy or args_dict.get("save_policy", ""),
        "runtime_environment": runtime_environment(),
    }
    metadata.update(git)
    if extra:
        metadata.update(dict(extra))
    return metadata


def write_metadata(path: str | os.PathLike[str], metadata: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
