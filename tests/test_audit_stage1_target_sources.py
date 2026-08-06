from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase2.audit_stage1_target_sources import write_audit_outputs


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_stage1_target_sources_reports_host_available_and_cini_blocked(tmp_path: Path) -> None:
    host_manifest = tmp_path / "host.csv"
    write_csv(
        host_manifest,
        ["benchmark", "task", "similarity_split", "sequence", "label", "family", "group", "id"],
        [
            {
                "benchmark": "hvue",
                "task": "hvue_human_host_tropism",
                "similarity_split": "train",
                "sequence": "ACGT",
                "label": "0",
                "family": "",
                "group": "primary_forget",
                "id": "host1",
            }
        ],
    )

    cini_raw_dir = tmp_path / "CINI"
    write_csv(cini_raw_dir / "train.csv", ["sequence", "label"], [{"sequence": "AAAA", "label": "0"}])
    write_csv(cini_raw_dir / "dev.csv", ["sequence", "label"], [{"sequence": "CCCC", "label": "1"}])
    write_csv(cini_raw_dir / "test.csv", ["sequence", "label"], [{"sequence": "GGGG", "label": "0"}])

    unified_manifest = tmp_path / "manifest.csv"
    write_csv(
        unified_manifest,
        ["benchmark", "task", "split", "sequence", "label", "family", "group", "id"],
        [
            {
                "benchmark": "hvue",
                "task": "hvue_human_virus_pathogenicity_cini",
                "split": "train",
                "sequence": "TTTT",
                "label": "1",
                "family": "mixed",
                "group": "primary_forget",
                "id": "cini1",
            }
        ],
    )

    out_json = tmp_path / "audit.json"
    args = argparse.Namespace(
        out_json=str(out_json),
        host_formal_manifest=str(host_manifest),
        host_task="hvue_human_host_tropism",
        host_split_column="similarity_split",
        host_split_type="cluster_disjoint",
        cini_raw_dir=str(cini_raw_dir),
        cini_unified_manifest=str(unified_manifest),
        cini_task="hvue_human_virus_pathogenicity_cini",
    )
    write_audit_outputs(args)

    report = json.loads(out_json.read_text())
    metadata = json.loads((tmp_path / "audit_metadata.json").read_text())
    assert report["host_formal_source"]["status"] == "available"
    assert report["host_formal_source"]["split_type"] == "cluster_disjoint"
    assert report["cini_sources"]["raw_split_dir"]["status"] == "blocked"
    assert report["cini_sources"]["raw_split_dir"]["all_columns"] == ["label", "sequence"]
    assert report["cini_sources"]["unified_manifest"]["status"] == "blocked"
    assert report["cini_sources"]["unified_manifest"]["family_values"] == {"mixed": 1}
    assert metadata["phase"] == "audit_stage1_target_sources"
    assert metadata["host_status"] == "available"
