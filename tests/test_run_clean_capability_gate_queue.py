from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.run_clean_capability_gate_queue import (
    base_only_manifest,
    mini_checkpoint_manifest,
    queue_metadata_path,
    write_queue_metadata,
    write_status,
)


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        resume=False,
        python="/usr/bin/python3",
        out_root=str(tmp_path / "clean_gate"),
        source_task7r_dir=str(tmp_path / "task7r"),
        source_queue_status=str(tmp_path / "task5b_v2" / "task7r8_5bv2_queue_status.json"),
        task5a_out_dir=str(tmp_path / "task5a"),
        task7_calibration=str(tmp_path / "task7" / "identity_capability_calibration.json"),
        stop_on_low_disk_gb=60.0,
        match_quantiles="1.0,0.75,0.50",
        smoke_layers="0,4,8,12,15",
        mini_layers="0,4,8,12,15",
        probe_seeds="42,43,44",
        c_grid="0.001,0.01,0.1,1.0",
        validity_c_grid="0.001,0.01,0.1,1.0,10.0",
        n_bootstrap=200,
        batch_size=4,
        device="cuda:0",
        cuda_visible_devices="0",
        model_dir="./evo-1-8k-base",
        config_path="configs/evo-1-8k-base_inference.yml",
        checkpoint_format="auto",
    )


def test_write_queue_metadata_and_status_include_provenance_paths(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    Path(args.out_root).mkdir(parents=True)

    write_queue_metadata(args)
    write_status(args, "running", started_at="2026-07-27T00:00:00Z")

    meta = json.loads(queue_metadata_path(args).read_text())
    status = json.loads((Path(args.out_root) / "clean_gate_queue_status.json").read_text())
    assert meta["phase"] == "clean_capability_gate_queue"
    assert meta["task"] == "clean_capability_gate_queue"
    assert meta["out_root"] == args.out_root
    assert status["queue_metadata_path"] == str(queue_metadata_path(args))
    assert status["status"] == "running"


def test_base_only_manifest_writes_companion_metadata(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    Path(args.out_root, "manifests").mkdir(parents=True)

    manifest_path = base_only_manifest(args)

    manifest = json.loads(manifest_path.read_text())
    meta = json.loads((Path(args.out_root) / "manifests" / "base_only_checkpoint_manifest_metadata.json").read_text())
    assert manifest["task"] == "clean_gate_base_only_manifest"
    assert manifest["checkpoints"][0]["checkpoint_name"] == "base"
    assert meta["phase"] == "clean_gate_base_only_manifest"
    assert meta["checkpoint_count"] == 1


def test_mini_checkpoint_manifest_writes_companion_metadata(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    task5a_dir = Path(args.task5a_out_dir)
    out_root = Path(args.out_root)
    task5a_dir.mkdir(parents=True)
    (out_root / "manifests").mkdir(parents=True)

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
                        "checkpoint_name": "gd_loc_s1000",
                        "method_family": "gd",
                        "checkpoint_path": "gd_loc_s1000.safetensors",
                        "checkpoint_exists": False,
                        "retain_safety_flag": "pass",
                    }
                ]
            }
        )
        + "\n"
    )

    manifest_path = mini_checkpoint_manifest(args)

    manifest = json.loads(manifest_path.read_text())
    meta = json.loads((out_root / "manifests" / "mini_task5b_checkpoint_manifest_metadata.json").read_text())
    assert manifest["task"] == "mini_task5b_checkpoint_manifest"
    assert meta["phase"] == "mini_task5b_checkpoint_manifest"
    assert meta["checkpoint_count"] == len(manifest["checkpoints"])
    assert "gd_loc_s1000" in meta["checkpoint_names"]
