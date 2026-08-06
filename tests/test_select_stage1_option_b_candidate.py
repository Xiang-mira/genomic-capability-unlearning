from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_select_stage1_option_b_candidate_picks_highest_test_auroc(tmp_path: Path) -> None:
    summary_a = tmp_path / "a.csv"
    summary_b = tmp_path / "b.csv"
    write_csv(
        summary_a,
        [
            {
                "config_id": "baseline",
                "status": "completed",
                "test_auroc_after_ascent": "0.42",
                "val_auroc_after_ascent": "0.45",
                "alpha_target": "1.0",
                "alpha_retain": "1.0",
                "elicitation_steps": "20",
                "ascent_steps": "20",
                "readout_disruption_flag": "readout_disruption",
                "weights_path": "baseline.safetensors",
            }
        ],
    )
    write_csv(
        summary_b,
        [
            {
                "config_id": "better",
                "status": "completed",
                "test_auroc_after_ascent": "0.44",
                "val_auroc_after_ascent": "0.47",
                "alpha_target": "1.0",
                "alpha_retain": "2.0",
                "elicitation_steps": "40",
                "ascent_steps": "40",
                "readout_disruption_flag": "readout_disruption",
                "weights_path": "better.safetensors",
            }
        ],
    )
    out_json = tmp_path / "best.json"
    subprocess.run(
        [
            sys.executable,
            "phase2/select_stage1_option_b_candidate.py",
            "--summary-csv",
            str(summary_a),
            "--summary-csv",
            str(summary_b),
            "--out-json",
            str(out_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    report = json.loads(out_json.read_text())
    metadata = json.loads((tmp_path / "best_metadata.json").read_text())
    assert report["best_candidate"]["config_id"] == "better"
    assert report["best_candidate"]["weights_path"] == "better.safetensors"
    assert report["ranked_candidates"][0]["config_id"] == "better"
    assert metadata["phase"] == "select_stage1_option_b_candidate"
    assert metadata["best_candidate_config_id"] == "better"
