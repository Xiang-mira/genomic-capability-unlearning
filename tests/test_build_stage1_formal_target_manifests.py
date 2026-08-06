from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase2.build_stage1_formal_target_manifests import build_formal_target_outputs


def test_build_stage1_formal_target_manifests_normalizes_host_manifest(tmp_path: Path) -> None:
    src = tmp_path / "host.csv"
    with src.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark", "task", "similarity_split", "sequence", "label", "group", "id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "benchmark": "hvue",
                "task": "hvue_human_host_tropism",
                "similarity_split": "train",
                "sequence": "ACGT",
                "label": "0",
                "group": "primary_forget",
                "id": "row1",
            }
        )
        writer.writerow(
            {
                "benchmark": "hvue",
                "task": "hvue_human_host_tropism",
                "similarity_split": "test",
                "sequence": "TGCA",
                "label": "1",
                "group": "primary_forget",
                "id": "row2",
            }
        )

    cini_raw_dir = tmp_path / "cini_raw"
    cini_raw_dir.mkdir()
    for split_name in ("train", "dev", "test"):
        with (cini_raw_dir / f"{split_name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sequence", "label"])
            writer.writeheader()
            writer.writerow({"sequence": "AAAA", "label": "0"})

    cini_unified = tmp_path / "cini_manifest.csv"
    with cini_unified.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark", "task", "split", "sequence", "label", "family", "group", "id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "benchmark": "hvue",
                "task": "hvue_human_virus_pathogenicity_cini",
                "split": "train",
                "sequence": "CCCC",
                "label": "1",
                "family": "mixed",
                "group": "primary_forget",
                "id": "cini1",
            }
        )

    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        out_dir=str(out_dir),
        host_formal_manifest=str(src),
        host_task="hvue_human_host_tropism",
        host_split_column="similarity_split",
        host_split_type="cluster_disjoint",
        host_group="primary_forget",
        cini_formal_manifest="",
        cini_task="hvue_human_virus_pathogenicity_cini",
        cini_split_column="split",
        cini_split_type="cluster_disjoint",
        cini_group="primary_forget",
        cini_raw_dir=str(cini_raw_dir),
        cini_unified_manifest=str(cini_unified),
    )
    build_formal_target_outputs(args)

    merged_path = out_dir / "stage1_formal_targets_available_manifest.csv"
    with merged_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["split_type"] == "cluster_disjoint"
    assert rows[0]["split"] == "train"

    report = json.loads((out_dir / "stage1_formal_target_manifest_report.json").read_text())
    metadata = json.loads((out_dir / "stage1_formal_target_manifest_metadata.json").read_text())
    assert report["missing_targets"][0]["task"] == "hvue_human_virus_pathogenicity_cini"
    assert report["source_audit"]["host_formal_source"]["status"] == "available"
    assert report["source_audit"]["cini_sources"]["raw_split_dir"]["status"] == "blocked"
    assert metadata["phase"] == "build_stage1_formal_target_manifests"
    assert metadata["merged_manifest"] == str(merged_path)
