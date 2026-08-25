#!/usr/bin/env python3
"""Group 1 preflight/status for A1 GENEB and G1-B inventory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]
SOURCE_BRANCH = "viral-benchmark-continuation"
SOURCE_COMMIT = "2cf5c967ce6739026e7fabd2381b394b5add4b64"
def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def geneb_data_status() -> dict[str, object]:
    root = Path(os.environ.get("VB_GENEB_DIR", os.environ.get("VB_OUT", str(ROOT / "results_viral_bench")) + "/geneb"))
    found = []
    if root.exists():
        for suffix in ("*.parquet", "*.csv"):
            for p in root.rglob(suffix):
                name = p.name.lower()
                if "train" in name or name in {"manifest.csv", "manifest.parquet"}:
                    found.append(str(p))
    found = sorted(found)
    manifest = Path(os.environ.get("VB_GENEB_TASK_MANIFEST", str(root / "sentinel_tasks.csv")))
    tasks = []
    if manifest.exists():
        with manifest.open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "task" in reader.fieldnames:
                tasks = [row["task"] for row in reader if row.get("task")]
    return {
        "root": str(root),
        "found_any": bool(found),
        "example_files": found[:20],
        "task_manifest": str(manifest),
        "task_manifest_exists": manifest.exists(),
        "task_count": len(tasks),
        "example_tasks": tasks[:20],
    }


def gpu_status() -> dict[str, object]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return {"available": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def status_payload() -> dict[str, object]:
    branch = run(["git", "branch", "--show-current"])
    head = run(["git", "rev-parse", "HEAD"])
    origin = run(["git", "rev-parse", "origin/viral-benchmark-continuation"])
    required = [
        "reports/PAPER_DESIGN.md",
        "reports/PROTOCOL.md",
        "reports/TESTED_MATRIX.md",
        "reports/RESEARCH_PLAN.md",
        "README.md",
        "scripts/common/capacity_sweep.py",
        "scripts/common/partial_overlap_audit.py",
        "scripts/common/build_strict_splits.py",
        "scripts/common/paired_bootstrap.py",
    ]
    missing = [p for p in required if not exists(p)]
    geneb = geneb_data_status()
    gpu = gpu_status()
    sbatch = shutil.which("sbatch") is not None
    return {
        "source_branch": SOURCE_BRANCH,
        "source_commit": SOURCE_COMMIT,
        "working_branch": branch,
        "working_commit": head,
        "origin_viral_benchmark_continuation": origin,
        "source_synced": SOURCE_COMMIT == origin,
        "required_paths_missing": missing,
        "geneb_data": geneb,
        "scheduler": {"sbatch": sbatch, "squeue": shutil.which("squeue") is not None},
        "gpu": gpu,
        "can_smoke_group1": not missing and geneb["found_any"] and gpu["available"],
        "can_submit_group1": not missing and geneb["found_any"] and geneb["task_count"] >= 13 and sbatch and gpu["available"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "submit"], default="submit")
    args = parser.parse_args()
    payload = status_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"source_branch: {payload['source_branch']}")
        print(f"source_commit: {payload['source_commit']}")
        print(f"working_branch: {payload['working_branch']}")
        print(f"working_commit: {payload['working_commit']}")
        print(f"origin_viral_benchmark_continuation: {payload['origin_viral_benchmark_continuation']}")
        print(f"can_smoke_group1: {payload['can_smoke_group1']}")
        print(f"can_submit_group1: {payload['can_submit_group1']}")
        if payload["required_paths_missing"]:
            print("missing required paths:")
            for p in payload["required_paths_missing"]:
                print(f"  - {p}")
        if not payload["geneb_data"]["found_any"]:
            print(f"GENEB data not found under {payload['geneb_data']['root']}")
        if payload["geneb_data"]["task_count"] < 13:
            print(
                "GENEB 13-task manifest not available: "
                f"{payload['geneb_data']['task_manifest']} "
                f"(task_count={payload['geneb_data']['task_count']})"
            )
        if not payload["scheduler"]["sbatch"]:
            print("Slurm sbatch not available")
        if not payload["gpu"]["available"]:
            print(f"GPU unavailable: {payload['gpu'].get('reason') or payload['gpu'].get('stderr') or payload['gpu'].get('stdout')}")
    if not args.preflight:
        return 0
    key = "can_smoke_group1" if args.mode == "smoke" else "can_submit_group1"
    return 0 if payload[key] else 1


if __name__ == "__main__":
    raise SystemExit(main())
