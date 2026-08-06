from __future__ import annotations

import argparse
from pathlib import Path

from phase2.preflight_route_decision import (
    relative_to_project,
    resolve_workspace_snapshot_dir,
    workspace_snapshot_command,
)


def test_relative_to_project_prefers_relative_path(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    nested = project_root / "data" / "snapshot"
    nested.mkdir(parents=True)
    assert relative_to_project(nested, project_root) == "data/snapshot"


def test_resolve_workspace_snapshot_dir_defaults_under_preflight(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    preflight_dir = project_root / "out" / "preflight"
    resolved = resolve_workspace_snapshot_dir(project_root, preflight_dir, "")
    assert resolved == preflight_dir / "workspace_state"


def test_workspace_snapshot_command_uses_python_and_relative_out_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    preflight_dir = project_root / "out" / "preflight"
    args = argparse.Namespace(
        python_bin="/envs/UT-p1/bin/python",
        workspace_snapshot_out_dir="data/custom_snapshot",
    )

    command, snapshot_dir = workspace_snapshot_command(
        args=args,
        project_root=project_root,
        preflight_dir=preflight_dir,
    )

    assert snapshot_dir == project_root / "data" / "custom_snapshot"
    assert command == [
        "/envs/UT-p1/bin/python",
        "-u",
        "phase2/freeze_workspace_state.py",
        "--project-root",
        str(project_root),
        "--out-dir",
        "data/custom_snapshot",
    ]
