from __future__ import annotations

import json
from pathlib import Path

from phase2.stage1_queue_supervisor import all_runs_complete, collect_progress_counts


def test_collect_progress_counts_reads_progress_files(tmp_path) -> None:
    out_root = tmp_path / "exp"
    run_dir = out_root / "fresh_lora/base/rank_8/lr_1e-5/seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "eval_benchmarks_progress.json").write_text(
        json.dumps({"status": "complete"})
    )
    run_dir2 = out_root / "fresh_lora/base/rank_8/lr_1e-5/seed_43"
    run_dir2.mkdir(parents=True)
    (run_dir2 / "eval_benchmarks_progress.json").write_text(
        json.dumps({"status": "running"})
    )

    counts = collect_progress_counts(out_root)

    assert counts == {"complete": 1, "running": 1}


def test_all_runs_complete_requires_expected_count(tmp_path) -> None:
    out_root = tmp_path / "exp"
    for seed in (42, 43):
        run_dir = out_root / f"fresh_lora/base/rank_8/lr_1e-5/seed_{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "eval_benchmarks_progress.json").write_text(
            json.dumps({"status": "complete"})
        )

    assert all_runs_complete(out_root, expected=2) is True
    assert all_runs_complete(out_root, expected=3) is False


def test_collect_progress_counts_supports_multiple_checkpoints(tmp_path) -> None:
    out_root = tmp_path / "exp"
    run_dir = out_root / "fresh_lora/projection_rank32/rank_16/lr_5e-5/seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "eval_benchmarks_progress.json").write_text(json.dumps({"status": "complete"}))

    counts = collect_progress_counts(out_root)

    assert counts == {"complete": 1}
