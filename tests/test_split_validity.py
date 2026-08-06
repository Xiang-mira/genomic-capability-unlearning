from __future__ import annotations

import subprocess
import csv
import json
import sys
from pathlib import Path

from phase2.check_split_validity import merge_rows, summarize_baselines, summarize_manifest


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_summary_flags_label_imbalance(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [
            {"task": "task_a", "split": "test", "split_type": "random", "label": "0"},
            {"task": "task_a", "split": "test", "split_type": "random", "label": "0"},
            {"task": "task_a", "split": "test", "split_type": "random", "label": "1"},
            {"task": "task_a", "split": "test", "split_type": "cluster_disjoint", "label": "0"},
            {"task": "task_a", "split": "test", "split_type": "cluster_disjoint", "label": "0"},
            {"task": "task_a", "split": "test", "split_type": "cluster_disjoint", "label": "0"},
        ],
    )

    rows = summarize_manifest(manifest, min_test_fraction=0.4)
    by_split = {row["split_type"]: row for row in rows}

    assert by_split["random"]["label_balance_ok"] is False
    assert by_split["cluster_disjoint"]["validity_note"] == "label_balance_fail"


def test_baseline_summary_requires_disjoint_gap(tmp_path: Path) -> None:
    baseline = tmp_path / "kmer.csv"
    write_csv(
        baseline,
        [
            {"task": "task_a", "split_type": "random", "auroc": 0.90},
            {"task": "task_a", "split_type": "cluster_disjoint", "auroc": 0.84},
            {"task": "task_b", "split_type": "random", "auroc": 0.70},
            {"task": "task_b", "split_type": "cluster_disjoint", "auroc": 0.69},
        ],
    )

    rows, _ = summarize_baselines([baseline], min_drop=0.02)
    by_task = {row["task"]: row for row in rows}

    assert by_task["task_a"]["disjoint_harder_ok"] is True
    assert by_task["task_b"]["validity_note"] == "baseline_gap_too_small"


def test_merge_rows_only_requires_gap_for_disjoint_splits() -> None:
    manifest_rows = [
        {
            "task": "task_a",
            "split_type": "random",
            "test_rows": 10,
            "n_labels": 2,
            "label_counts": json.dumps({"0": 5, "1": 5}),
            "minority_fraction": 0.5,
            "label_balance_ok": True,
            "validity_note": "",
        },
        {
            "task": "task_a",
            "split_type": "cluster_disjoint",
            "test_rows": 10,
            "n_labels": 2,
            "label_counts": json.dumps({"0": 5, "1": 5}),
            "minority_fraction": 0.5,
            "label_balance_ok": True,
            "validity_note": "",
        },
    ]
    baseline_rows = [
        {
            "task": "task_a",
            "random_score": 0.9,
            "cluster_disjoint_score": 0.85,
            "random_minus_disjoint": 0.05,
            "disjoint_harder_ok": True,
            "validity_note": "",
        }
    ]

    merged = {row["split_type"]: row for row in merge_rows(manifest_rows, baseline_rows)}
    assert merged["random"]["overall_ok"] is True
    assert merged["cluster_disjoint"]["overall_ok"] is True


def test_check_split_validity_writes_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    baseline = tmp_path / "baseline.csv"
    out_csv = tmp_path / "validity.csv"
    write_csv(
        manifest,
        [
            {"task": "task_a", "split": "test", "split_type": "cluster_disjoint", "label": "0"},
            {"task": "task_a", "split": "test", "split_type": "cluster_disjoint", "label": "1"},
        ],
    )
    write_csv(
        baseline,
        [
            {"task": "task_a", "split_type": "random", "auroc": 0.90},
            {"task": "task_a", "split_type": "cluster_disjoint", "auroc": 0.80},
        ],
    )
    subprocess.run(
        [
            sys.executable,
            "phase2/check_split_validity.py",
            "--manifest",
            str(manifest),
            "--baseline-csv",
            str(baseline),
            "--out-csv",
            str(out_csv),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    metadata = json.loads((tmp_path / "validity_metadata.json").read_text())
    assert metadata["phase"] == "check_split_validity"
    assert metadata["manifest_tasks"] == ["task_a"]
