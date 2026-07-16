"""Read-only storage and artifact inventory for Task 0.

This script deliberately does not delete, move, compress, or rewrite existing
checkpoints. It only writes audit reports under the requested output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


REAUDIT_REQUIRED_SUFFIXES = {
    "data/phase2/checkpoints_projection_opt/projopt_host5_9_coro4_10_coro125",
    "data/phase2/checkpoints_projection_adaptive_rank16/projopt_host5_9_coro0_10_adaptive_basis_rank16",
    "data/phase2/checkpoints_projection_adaptive_rank32/projopt_host5_9_coro0_10_adaptive_basis_rank32",
    "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000",
    "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s500",
    "data/phase2/checkpoints_tuned/refseq_gd_projinit_full_ar5_s200",
    "data/phase2/checkpoints_tuned/refseq_gd_projinit_random_ar5_s1000",
    "data/phase2/checkpoints_rmu_localized_joint_probe/rmu_loc_l5_l9_jointprobe_sc200_ar5_s500",
    "data/phase2/checkpoints_rmu_localized_joint_probe/rmu_loc_l5_l9_jointprobe_sc50_ar5_s500",
    "data/phase2/checkpoints_rmu_localized_joint_probe/rmu_loc_l5_l9_jointprobe_sc100_ar5_s500",
    "data/phase2/checkpoints_rmu_pareto/rmu_pareto_l8_ratio050",
    "data/phase2/checkpoints_rmu_tuning/rmu_full_l6_base",
    "data/phase2/checkpoints_rmu_tuning/rmu_full_l8_base",
}

ARTIFACT_NAMES = {
    "weights.safetensors",
    "eval_auroc.csv",
    "eval_ppl.json",
    "eval_benchmarks.csv",
    "eval_benchmarks_summary.json",
    "meta.json",
    "log.json",
    "adaptive_basis_summary.json",
    "ids.npy",
    "labels.npy",
}


def rel(path: Path) -> str:
    return path.as_posix().lstrip("./")


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


def bytes_to_gb(value: int) -> float:
    return value / (1024**3)


def run_command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def disk_status(path: Path) -> dict[str, object]:
    usage = shutil.disk_usage(path)
    available_gb = bytes_to_gb(usage.free)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "available_bytes": usage.free,
        "total_gb": bytes_to_gb(usage.total),
        "used_gb": bytes_to_gb(usage.used),
        "available_gb": available_gb,
        "task0_3_allowed": available_gb >= 80,
        "future_delta_diagnostic_allowed": available_gb >= 80,
        "future_full_checkpoint_safety_margin": available_gb >= 120,
        "full_checkpoint_training_allowed": available_gb >= 120,
        "disk_status": "healthy" if available_gb >= 120 else "tight_but_usable" if available_gb >= 80 else "too_low",
    }


def classify_checkpoint(path: Path) -> str:
    path_rel = rel(path)
    if path_rel in REAUDIT_REQUIRED_SUFFIXES:
        return "needed_for_next_reaudit_or_init"
    if "adaptive_probe_bases" in path_rel:
        return "needed_for_next_reaudit_or_init"
    return "evidence_or_context_only"


def checkpoint_like_dirs(root: Path) -> list[Path]:
    dirs = set()
    if not root.exists():
        return []
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if item.name in ARTIFACT_NAMES or item.name.endswith(".log"):
            parent = item.parent
            dirs.add(parent)
            if item.name == "weights.safetensors":
                dirs.add(parent)
    return sorted(dirs)


def summarize_checkpoint_dir(path: Path) -> dict[str, object]:
    weights = path / "weights.safetensors"
    return {
        "path": rel(path),
        "category": classify_checkpoint(path),
        "size_bytes": dir_size(path),
        "size_gb": bytes_to_gb(dir_size(path)),
        "has_weights": weights.exists(),
        "weights_bytes": file_size(weights),
        "weights_gb": bytes_to_gb(file_size(weights)),
        "has_eval_auroc": (path / "eval_auroc.csv").exists(),
        "has_eval_ppl": (path / "eval_ppl.json").exists(),
        "has_eval_benchmarks_csv": (path / "eval_benchmarks.csv").exists(),
        "has_eval_benchmarks_json": (path / "eval_benchmarks_summary.json").exists(),
        "has_meta": (path / "meta.json").exists(),
        "has_log_json": (path / "log.json").exists(),
    }


def artifact_rows(roots: Iterable[Path]) -> list[dict[str, object]]:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if (
                path.name in ARTIFACT_NAMES
                or path.name.endswith(".log")
                or path.suffix in {".csv", ".json", ".npz", ".npy"}
            ):
                rows.append(
                    {
                        "path": rel(path),
                        "parent": rel(path.parent),
                        "name": path.name,
                        "size_bytes": file_size(path),
                        "size_gb": bytes_to_gb(file_size(path)),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-root", default="data/phase2")
    parser.add_argument("--family-root", default="data/family_targets")
    parser.add_argument("--host-root", default="data/host_tropism")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--out-dir", default="data/phase2/audits/task0_3_20260713")
    args = parser.parse_args()

    phase2_root = Path(args.phase2_root)
    roots = [phase2_root, Path(args.family_root), Path(args.host_root), Path(args.logs_root)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disk = disk_status(Path("."))
    phase2_children = []
    if phase2_root.exists():
        for child in sorted(item for item in phase2_root.iterdir() if item.is_dir()):
            size = dir_size(child)
            phase2_children.append({"path": rel(child), "size_bytes": size, "size_gb": bytes_to_gb(size)})

    weights = sorted(phase2_root.rglob("weights.safetensors")) if phase2_root.exists() else []
    weights_total = sum(file_size(path) for path in weights)
    checkpoints = [summarize_checkpoint_dir(path) for path in checkpoint_like_dirs(phase2_root)]
    artifacts = artifact_rows(roots)

    git_status = run_command(["git", "status", "--short"])
    git_payload = {
        "commit_hash": run_command(["git", "rev-parse", "HEAD"]),
        "commit_subject": run_command(["git", "log", "-1", "--pretty=%s"]),
        "status_short": git_status.splitlines() if git_status else [],
    }

    inventory = {
        "disk": disk,
        "phase2_root": rel(phase2_root),
        "phase2_root_size_bytes": dir_size(phase2_root) if phase2_root.exists() else 0,
        "phase2_root_size_gb": bytes_to_gb(dir_size(phase2_root)) if phase2_root.exists() else 0,
        "phase2_children": phase2_children,
        "weights_safetensors_count": len(weights),
        "weights_safetensors_total_bytes": weights_total,
        "weights_safetensors_total_gb": bytes_to_gb(weights_total),
        "checkpoints": checkpoints,
        "git": git_payload,
        "notes": [
            "Read-only inventory: no deletion, migration, compression, or checkpoint modification was performed.",
            "Checkpoint categories are limited to needed_for_next_reaudit_or_init and evidence_or_context_only.",
        ],
    }

    (out_dir / "storage_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    write_csv(out_dir / "checkpoint_manifest.csv", checkpoints)
    write_csv(out_dir / "artifact_manifest.csv", artifacts)
    (out_dir / "git_status.txt").write_text(
        f"commit_hash: {git_payload['commit_hash']}\n"
        f"commit_subject: {git_payload['commit_subject']}\n\n"
        + "\n".join(git_payload["status_short"])
        + ("\n" if git_payload["status_short"] else "")
    )

    enough = "YES" if disk["task0_3_allowed"] else "NO"
    full_enough = "YES" if disk["future_full_checkpoint_safety_margin"] else "NO"
    summary = f"""# Task 0 Storage State

## Files Produced

- storage_inventory.json
- checkpoint_manifest.csv
- artifact_manifest.csv
- current_state_summary.md
- git_status.txt

## Disk Gate

- Available GB: {disk['available_gb']:.2f}
- Task 0-3 allowed: {enough}
- Disk status: {disk['disk_status']}
- Full checkpoint safety margin >= 120G: {full_enough}

## Decision

1. The audit files listed above were produced.
2. The current free space is {'enough' if disk['task0_3_allowed'] else 'not enough'} for Task 0-3.
3. The current free space is {'enough' if disk['future_full_checkpoint_safety_margin'] else 'not enough'} for default full-checkpoint training.
4. Next step: {'continue to Task 1' if disk['task0_3_allowed'] else 'pause training and only continue code/audit work'}.

No files were deleted, moved, compressed, or modified outside this audit output directory.
"""
    (out_dir / "current_state_summary.md").write_text(summary)
    print(f"[storage-audit] wrote reports to {out_dir}")
    print(f"[storage-audit] available={disk['available_gb']:.2f}G status={disk['disk_status']}")


if __name__ == "__main__":
    main()
