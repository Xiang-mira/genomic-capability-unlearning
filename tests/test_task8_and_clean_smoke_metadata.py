from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])


def test_run_task8_writes_meta(tmp_path: Path) -> None:
    task7_dir = tmp_path / "task7"
    out_dir = tmp_path / "task8"
    task7_dir.mkdir()
    out_dir.mkdir()
    task5a_summary = tmp_path / "task5a_summary.json"

    (task7_dir / "identity_capability_calibration.json").write_text(
        json.dumps(
            {
                "decision": {
                    "capability_probe_status": "clean_formal_gate",
                    "formal_success_allowed": True,
                }
            }
        )
        + "\n"
    )
    (task7_dir / "capability_probe_summary.csv").write_text(
        "model_name,hidden_incremental_auroc_mean,test_separability_mean\n"
        "hidden_only_model,0.06,0.70\n"
        "raw_hidden_joint_model,0.07,0.71\n"
        "family_hidden_joint_model,0.05,0.69\n"
        "raw_only_model,,0.52\n"
        "family_only_model,,0.50\n"
        "kmer_only_model,,0.48\n"
    )
    task5a_summary.write_text(json.dumps({"rows": [{"checkpoint_name": "base"}]}) + "\n")

    subprocess.run(
        [
            sys.executable,
            "phase2/run_task8_identity_capability_calibration.py",
            "--task7-dir",
            str(task7_dir),
            "--task5a-summary",
            str(task5a_summary),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    meta = json.loads((out_dir / "meta.json").read_text())
    payload = json.loads((out_dir / "identity_capability_calibration.json").read_text())
    assert meta["phase"] == "task8_identity_capability_calibration"
    assert meta["task"] == "task8_identity_capability_calibration"
    assert meta["relationship_case"] == payload["decision"]["relationship_case"]


def test_summarize_clean_smoke_writes_summary_metadata(tmp_path: Path) -> None:
    candidate_index = tmp_path / "candidate_index.json"
    smoke_root = tmp_path / "smoke"
    out_dir = tmp_path / "summary"
    smoke_dir = smoke_root / "candidate_q100"
    candidate_dir = tmp_path / "candidate_q100"
    (candidate_dir / "probe_validity").mkdir(parents=True)
    smoke_dir.mkdir(parents=True)
    out_dir.mkdir()

    candidate_index.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_name": "candidate_q100",
                        "candidate_quantile": 1.0,
                        "rows": 10,
                        "candidate_dir": str(candidate_dir),
                    }
                ]
            }
        )
        + "\n"
    )
    (candidate_dir / "probe_validity" / "probe_validity_audit.json").write_text(
        json.dumps({"decision": {"action": "continue"}}) + "\n"
    )
    (smoke_dir / "identity_capability_calibration.json").write_text(
        json.dumps(
            {
                "decision": {
                    "capability_probe_status": "clean_formal_gate",
                    "hidden_mean_separability": 0.50,
                    "shortcut_best_mean_separability": 0.52,
                },
                "raw_hidden_joint_model": {"hidden_incremental_auroc_mean": 0.04},
                "raw_plus_kmer_plus_metadata_hidden_joint_model": {"hidden_incremental_auroc_mean": 0.06},
                "hidden_only_model": {"hidden_incremental_auroc_mean": 0.05},
            }
        )
        + "\n"
    )
    (smoke_dir / "capability_probe_metrics.csv").write_text(
        "model_name,seed,hidden_incremental_auroc\n"
        "raw_hidden_joint_model,42,0.04\n"
        "raw_hidden_joint_model,43,0.03\n"
        "raw_plus_kmer_plus_metadata_hidden_joint_model,42,0.06\n"
        "raw_plus_kmer_plus_metadata_hidden_joint_model,43,0.05\n"
        "raw_plus_kmer_plus_metadata_hidden_joint_model,44,0.04\n"
        "hidden_only_model,42,0.05\n"
    )

    subprocess.run(
        [
            sys.executable,
            "phase2/summarize_clean_capability_gate_smoke.py",
            "--candidate-index",
            str(candidate_index),
            "--smoke-root",
            str(smoke_root),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    meta = json.loads((out_dir / "summary_metadata.json").read_text())
    summary = json.loads((out_dir / "clean_gate_smoke_summary.json").read_text())
    assert meta["phase"] == "clean_gate_smoke_summary"
    assert meta["task"] == "clean_capability_gate_smoke_summary"
    assert meta["selected_candidate_name"] == summary["selected_candidate"]["candidate_name"]


def test_build_clean_candidates_writes_build_and_candidate_meta(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.csv"
    out_root = tmp_path / "candidates"
    source_manifest.write_text(
        "sequence,label,split,task,benchmark,group,source_family,similarity_cluster_id\n"
        "AAAA,1,train,task,bench,g,famA,c1\n"
        "AAAT,1,val,task,bench,g,famA,c2\n"
        "TTTT,0,train,task,bench,g,famB,c1\n"
        "TTTA,0,val,task,bench,g,famB,c2\n"
    )

    subprocess.run(
        [
            sys.executable,
            "phase2/build_clean_capability_candidates.py",
            "--source-manifest",
            str(source_manifest),
            "--out-root",
            str(out_root),
            "--quantiles",
            "1.0",
            "--seeds",
            "42",
            "--c-grid",
            "0.1",
            "--n-bootstrap",
            "10",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    build_meta = json.loads((out_root / "candidate_build_metadata.json").read_text())
    candidate_index = json.loads((out_root / "candidate_index.json").read_text())
    candidate_dir = Path(candidate_index["candidates"][0]["candidate_dir"])
    candidate_meta = json.loads((candidate_dir / "meta.json").read_text())
    assert build_meta["phase"] == "clean_capability_candidate_build"
    assert build_meta["candidate_count"] == 1
    assert candidate_meta["phase"] == "clean_capability_candidate"
    assert candidate_meta["candidate_name"] == candidate_index["candidates"][0]["candidate_name"]
