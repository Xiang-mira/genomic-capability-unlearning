from __future__ import annotations

import json
import os
from pathlib import Path

from phase2.freeze_workspace_state import build_workspace_snapshot, latest_report, markdown_summary


def test_latest_report_picks_most_recent_report_like_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    older = docs / "old_report.md"
    newer = docs / "recent_summary.json"
    older.write_text("old\n")
    newer.write_text("{}\n")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    payload = latest_report([docs])
    assert payload["exists"] is True
    assert payload["path"] == str(newer)


def test_latest_report_excludes_snapshot_output_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    docs = tmp_path / "docs"
    snapshot_dir = data_root / "audits" / "workspace_state_20260727"
    docs.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    real_report = docs / "real_report.md"
    generated = snapshot_dir / "workspace_state_summary.md"
    real_report.write_text("real\n")
    generated.write_text("generated\n")
    os.utime(real_report, (1, 1))
    os.utime(generated, (10, 10))

    payload = latest_report([tmp_path], exclude_roots=[snapshot_dir])
    assert payload["exists"] is True
    assert payload["path"] == str(real_report)


def test_build_workspace_snapshot_captures_git_and_environment(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    docs = project_root / "docs"
    docs.mkdir(parents=True)
    report = docs / "current_report.md"
    report.write_text("report\n")

    outputs = {
        ("branch", "--show-current"): "codex/test\n",
        ("rev-parse", "HEAD"): "abc123\n",
        ("status", "--short"): " M phase2/foo.py\n?? tests/bar.py\n",
        ("diff", "--stat"): " phase2/foo.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n",
        ("diff",): "diff --git a/phase2/foo.py b/phase2/foo.py\n",
        ("diff", "--cached"): "",
    }

    def fake_git_output(cwd: Path, *args: str) -> str:
        assert cwd == project_root
        return outputs.get(tuple(args), "")

    monkeypatch.setattr("phase2.freeze_workspace_state.git_output", fake_git_output)
    monkeypatch.setattr(
        "phase2.freeze_workspace_state.runtime_environment",
        lambda: {"python": "3.13", "cwd": str(project_root)},
    )
    monkeypatch.setattr(
        "phase2.freeze_workspace_state.key_package_versions",
        lambda packages: {"torch": "2.0.0", "numpy": "2.1.0"},
    )

    snapshot = build_workspace_snapshot(
        project_root=project_root,
        report_roots=[docs],
        package_names=["torch", "numpy"],
        exclude_roots=[],
    )

    assert snapshot["current_branch"] == "codex/test"
    assert snapshot["commit_hash"] == "abc123"
    assert snapshot["git_status_short"] == [" M phase2/foo.py", "?? tests/bar.py"]
    assert snapshot["git_diff_present"] is True
    assert snapshot["runtime_environment"]["cwd"] == str(project_root)
    assert snapshot["key_package_versions"]["torch"] == "2.0.0"


def test_main_like_outputs_can_be_written(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    snapshot = {
        "created_at": "2026-07-27T00:00:00+00:00",
        "runtime_environment": {"python": "3.13"},
        "key_package_versions": {"torch": "2.0.0"},
    }
    (out_dir / "environment_report.json").write_text(
        json.dumps(
            {
                "created_at": snapshot["created_at"],
                "runtime_environment": snapshot["runtime_environment"],
                "key_package_versions": snapshot["key_package_versions"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    payload = json.loads((out_dir / "environment_report.json").read_text())
    assert payload["runtime_environment"]["python"] == "3.13"


def test_markdown_summary_includes_core_snapshot_fields() -> None:
    text = markdown_summary(
        {
            "created_at": "2026-07-27T00:00:00+00:00",
            "project_root": "/repo",
            "current_branch": "main",
            "commit_hash": "abc123",
            "git_status_short": [" M phase2/foo.py"],
            "git_diff_stat_lines": [" phase2/foo.py | 2 +-"],
            "git_diff_present": True,
            "unstaged_diff_line_count": 10,
            "staged_diff_line_count": 0,
            "latest_report": {"path": "/repo/data/report.md", "modified_at": "2026-07-27T01:00:00+00:00"},
            "key_package_versions": {"numpy": "2.1.0", "torch": "2.0.0"},
        }
    )
    assert "# Workspace State Snapshot" in text
    assert "- current_branch: main" in text
    assert "- commit_hash: abc123" in text
    assert "## Latest Report" in text
    assert "- torch: 2.0.0" in text
