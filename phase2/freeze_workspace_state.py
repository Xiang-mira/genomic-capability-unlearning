"""Freeze the current dirty workspace state for reproducibility audits.

This script is intentionally read-only with respect to the repo state. It
captures the current git/workspace snapshot and environment details into an
output directory so later experiment differences can be attributed cleanly.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.run_metadata import build_run_metadata, runtime_environment, write_metadata


REPORT_SUFFIXES = {".md", ".json", ".csv", ".txt"}
DEFAULT_REPORT_ROOTS = [
    "data/phase2",
    "docs",
]
DEFAULT_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "safetensors",
    "datasets",
    "huggingface_hub",
]


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def git_output(cwd: Path, *args: str) -> str:
    result = run_command(["git", *args], cwd)
    if result.returncode != 0:
        return ""
    return result.stdout


def key_package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in packages:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            versions[name] = f"unavailable: {exc.__class__.__name__}"
            continue
        version = getattr(module, "__version__", None)
        versions[name] = str(version) if version is not None else "unknown"
    return versions


def latest_report(report_roots: list[Path], exclude_roots: list[Path] | None = None) -> dict[str, Any]:
    excluded = [path.resolve() for path in (exclude_roots or [])]
    candidates: list[Path] = []
    for root in report_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if any(resolved == excluded_root or excluded_root in resolved.parents for excluded_root in excluded):
                continue
            if path.suffix.lower() not in REPORT_SUFFIXES:
                continue
            if "report" in path.name.lower() or "summary" in path.name.lower() or path.suffix.lower() == ".md":
                candidates.append(path)
    if not candidates:
        return {"path": "", "modified_at": "", "exists": False}
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return {
        "path": str(latest),
        "modified_at": now_from_timestamp(latest.stat().st_mtime),
        "exists": True,
    }


def now_from_timestamp(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def build_workspace_snapshot(
    *,
    project_root: Path,
    report_roots: list[Path],
    package_names: list[str],
    exclude_roots: list[Path] | None = None,
) -> dict[str, Any]:
    branch = git_output(project_root, "branch", "--show-current").strip()
    commit_hash = git_output(project_root, "rev-parse", "HEAD").strip()
    status_short = git_output(project_root, "status", "--short")
    diff_stat = git_output(project_root, "diff", "--stat")
    diff_patch = git_output(project_root, "diff")
    cached_diff_patch = git_output(project_root, "diff", "--cached")
    env_report = runtime_environment()
    package_versions = key_package_versions(package_names)
    latest = latest_report(report_roots, exclude_roots=exclude_roots)
    return {
        "created_at": now(),
        "project_root": str(project_root),
        "current_branch": branch,
        "commit_hash": commit_hash,
        "git_status_short": status_short.splitlines(),
        "git_diff_stat_lines": diff_stat.splitlines(),
        "git_diff_present": bool(diff_patch.strip() or cached_diff_patch.strip()),
        "unstaged_diff_line_count": len(diff_patch.splitlines()),
        "staged_diff_line_count": len(cached_diff_patch.splitlines()),
        "latest_report": latest,
        "runtime_environment": env_report,
        "key_package_versions": package_versions,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def markdown_summary(snapshot: dict[str, Any]) -> str:
    latest = snapshot.get("latest_report", {})
    package_versions = snapshot.get("key_package_versions", {})
    package_lines = [
        f"- {name}: {version}"
        for name, version in sorted(package_versions.items())
    ]
    git_status_lines = snapshot.get("git_status_short", [])
    diff_stat_lines = snapshot.get("git_diff_stat_lines", [])
    return "\n".join(
        [
            "# Workspace State Snapshot",
            "",
            f"- created_at: {snapshot.get('created_at', '')}",
            f"- project_root: {snapshot.get('project_root', '')}",
            f"- current_branch: {snapshot.get('current_branch', '')}",
            f"- commit_hash: {snapshot.get('commit_hash', '')}",
            f"- git_status_entries: {len(git_status_lines)}",
            f"- git_diff_present: {snapshot.get('git_diff_present', False)}",
            f"- unstaged_diff_line_count: {snapshot.get('unstaged_diff_line_count', 0)}",
            f"- staged_diff_line_count: {snapshot.get('staged_diff_line_count', 0)}",
            "",
            "## Latest Report",
            "",
            f"- path: {latest.get('path', '')}",
            f"- modified_at: {latest.get('modified_at', '')}",
            "",
            "## Key Packages",
            "",
            *(package_lines or ["- none"]),
            "",
            "## Git Status",
            "",
            *(git_status_lines or ["(clean)"]),
            "",
            "## Git Diff Stat",
            "",
            *(diff_stat_lines or ["(no diff stat)"]),
            "",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-dir", default="data/phase2/audits/workspace_state_20260727")
    parser.add_argument("--report-roots", default=",".join(DEFAULT_REPORT_ROOTS))
    parser.add_argument("--packages", default=",".join(DEFAULT_PACKAGES))
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_roots = [(project_root / item.strip()).resolve() for item in args.report_roots.split(",") if item.strip()]
    packages = [item.strip() for item in args.packages.split(",") if item.strip()]

    snapshot = build_workspace_snapshot(
        project_root=project_root,
        report_roots=report_roots,
        package_names=packages,
        exclude_roots=[out_dir],
    )

    git_status_text = git_output(project_root, "status", "--short")
    git_diff_stat_text = git_output(project_root, "diff", "--stat")
    git_diff_text = git_output(project_root, "diff")
    git_cached_diff_text = git_output(project_root, "diff", "--cached")

    write_text(out_dir / "git_status.txt", git_status_text)
    write_text(out_dir / "git_diff_stat.txt", git_diff_stat_text)
    write_text(out_dir / "git_diff.patch", git_diff_text)
    write_text(out_dir / "git_diff_cached.patch", git_cached_diff_text)
    write_text(
        out_dir / "environment_report.json",
        json.dumps(
            {
                "created_at": snapshot["created_at"],
                "runtime_environment": snapshot["runtime_environment"],
                "key_package_versions": snapshot["key_package_versions"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(out_dir / "workspace_state_snapshot.json", json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    write_text(out_dir / "workspace_state_summary.md", markdown_summary(snapshot))
    captured_files = [
        "git_status.txt",
        "git_diff_stat.txt",
        "git_diff.patch",
        "git_diff_cached.patch",
        "environment_report.json",
        "workspace_state_snapshot.json",
        "workspace_state_summary.md",
    ]
    write_metadata(
        out_dir / "workspace_state_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[str(out_dir / name) for name in captured_files],
            extra={
                "phase": "freeze_workspace_state",
                "task": "workspace_state_snapshot",
                "project_root": str(project_root),
                "out_dir": str(out_dir),
                "current_branch": snapshot["current_branch"],
                "commit_hash_snapshot": snapshot["commit_hash"],
                "latest_report": snapshot["latest_report"],
                "report_roots": [str(root) for root in report_roots],
                "captured_files": captured_files,
            },
        ),
    )
    print(f"[freeze-workspace] wrote snapshot to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
