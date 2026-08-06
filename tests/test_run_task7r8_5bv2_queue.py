from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.run_task7r8_5bv2_queue import (
    build_task5b_v2_manifest,
    queue_metadata_path,
    write_queue_metadata,
    write_status,
)


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        task7r_out_dir=str(tmp_path / "task7r"),
        task8_out_dir=str(tmp_path / "task8"),
        task5b_v2_out_dir=str(tmp_path / "task5b_v2"),
        queue_status=str(tmp_path / "task5b_v2" / "task7r8_5bv2_queue_status.json"),
        task5a_out_dir=str(tmp_path / "task5a"),
        task5a_manifest=str(tmp_path / "task5a" / "task5a_for_task7_checkpoint_manifest.json"),
        benchmark_manifest=str(tmp_path / "bench.csv"),
        primary_task="hvue_human_transmissibility_coronaviridae",
        aux_task="hvue_human_host_tropism",
        formal_split_column="similarity_split",
        max_per_split_label=400,
        n_bootstrap=200,
        layers="0-15",
        batch_size=4,
        device="cuda:0",
        model_dir="./evo-1-8k-base",
        config_path="configs/evo-1-8k-base_inference.yml",
        checkpoint_format="auto",
        probe_seeds="42,43,44",
        fresh_c_grid="0.001,0.01,0.1,1.0",
        stop_on_low_disk_gb=60.0,
    )


def test_write_queue_metadata_and_status_include_provenance_paths(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    Path(args.task5b_v2_out_dir).mkdir(parents=True)

    write_queue_metadata(args)
    write_status(args, "started")

    meta = json.loads(queue_metadata_path(args).read_text())
    status = json.loads(Path(args.queue_status).read_text())
    assert meta["phase"] == "task7r8_5bv2_queue"
    assert meta["task"] == "task7r8_5bv2_queue"
    assert meta["queue_status_path"] == args.queue_status
    assert status["queue_metadata_path"] == str(queue_metadata_path(args))
    assert status["status"] == "started"


def test_build_task5b_v2_manifest_writes_companion_metadata(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    task5a_dir = Path(args.task5a_out_dir)
    task5b_dir = Path(args.task5b_v2_out_dir)
    task5a_dir.mkdir(parents=True)
    task5b_dir.mkdir(parents=True)

    Path(args.task5a_manifest).write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_name": "base",
                        "source_checkpoint_name": "base",
                        "method_family": "base",
                        "checkpoint_path": "",
                        "checkpoint_exists": True,
                    },
                    {
                        "checkpoint_name": "best_gd_from_task5a",
                        "source_checkpoint_name": "gd_full_control",
                        "method_family": "gd",
                        "checkpoint_path": "gd_full.safetensors",
                        "checkpoint_exists": False,
                    },
                ]
            }
        )
        + "\n"
    )

    manifest_path = build_task5b_v2_manifest(args)

    manifest = json.loads(manifest_path.read_text())
    meta = json.loads((task5b_dir / "task5b_v2_checkpoint_manifest_metadata.json").read_text())
    assert manifest["task"] == "task5b_v2_clean_probe_checkpoint_manifest"
    assert meta["phase"] == "task5b_v2_checkpoint_manifest"
    assert meta["checkpoint_count"] >= 2
    assert "base" in meta["checkpoint_names"]
