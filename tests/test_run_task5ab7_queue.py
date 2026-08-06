from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.run_task5ab7_queue import build_task5b_manifest, queue_metadata_path, write_queue_metadata, write_queue_status


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        task5a_out_dir=str(tmp_path / "task5a"),
        task7_out_dir=str(tmp_path / "task7"),
        task5b_out_dir=str(tmp_path / "task5b"),
        config_path="configs/evo-1-8k-base_inference.yml",
        task7_ready_flag=str(tmp_path / "task7" / "task7_code_ready.flag"),
        model_dir="./evo-1-8k-base",
        stop_on_low_disk_gb=80.0,
        wait_for_task7_ready=False,
        ready_poll_seconds=300,
        retry_on_failure=1,
        oom_retry_batch_size=2,
        batch_size=4,
        max_eval=400,
        layers="0-15",
        probe_seeds="42,43,44",
        fresh_c_grid="0.001,0.01,0.1,1.0",
        n_bootstrap=200,
        device="cuda:0",
        cuda_visible_devices="0",
        checkpoint_format="auto",
    )


def test_write_queue_metadata_and_status_include_provenance_paths(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    Path(args.task5b_out_dir).mkdir(parents=True)

    write_queue_metadata(args)
    write_queue_status(args, "running", started_at="2026-07-27T00:00:00Z")

    meta = json.loads(queue_metadata_path(args).read_text())
    status = json.loads((Path(args.task5b_out_dir) / "task5ab7_queue_status.json").read_text())
    assert meta["phase"] == "task5ab7_queue"
    assert meta["task"] == "task5ab7_queue"
    assert meta["task5b_out_dir"] == args.task5b_out_dir
    assert status["queue_metadata_path"] == str(queue_metadata_path(args))
    assert status["status"] == "running"


def test_build_task5b_manifest_writes_companion_metadata(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    task5a_dir = Path(args.task5a_out_dir)
    task5b_dir = Path(args.task5b_out_dir)
    task5a_dir.mkdir(parents=True)
    task5b_dir.mkdir(parents=True)

    (task5a_dir / "task5a_for_task7_checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_name": "base",
                        "source_checkpoint_name": "base",
                        "method_family": "base",
                        "checkpoint_path": "",
                        "checkpoint_exists": True,
                    }
                ]
            }
        )
        + "\n"
    )
    (task5a_dir / "task5a_identity_reaudit_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "checkpoint_name": "projection_rank16",
                        "method_family": "projection",
                        "checkpoint_path": "projection_rank16.safetensors",
                        "checkpoint_exists": False,
                        "run_status": "completed",
                        "recommended_for_capability_reaudit": True,
                        "recommended_for_p5_init": False,
                        "retain_safety_flag": "pass",
                        "fresh_family_mean_separability": 0.7,
                        "fresh_family_max_separability": 0.8,
                    }
                ]
            }
        )
        + "\n"
    )

    manifest_path = build_task5b_manifest(args)

    manifest = json.loads(manifest_path.read_text())
    meta = json.loads((task5b_dir / "task5b_checkpoint_manifest_metadata.json").read_text())
    assert manifest["task"] == "task5b_capability_reaudit_checkpoint_manifest"
    assert len(manifest["checkpoints"]) == 2
    assert meta["phase"] == "task5b_checkpoint_manifest"
    assert meta["checkpoint_count"] == 2
    assert "projection_rank16" in meta["checkpoint_names"]
