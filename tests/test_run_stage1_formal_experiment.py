from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.run_stage1_formal_experiment import (
    available_formal_tasks,
    build_lora_commands,
    build_probe_vs_sft_commands,
    checkpoint_specs,
    load_kmer_baseline_map,
    resolve_option_b_checkpoint,
)


def test_available_formal_tasks_reads_blocked_targets(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "manifests": {
                    "hvue_human_host_tropism": {"path": "host.csv"},
                },
                "missing_targets": [{"task": "hvue_human_virus_pathogenicity_cini"}],
                "merged_manifest": "merged.csv",
            }
        )
    )

    tasks, blocked, merged_manifest = available_formal_tasks(report)

    assert tasks == ["hvue_human_host_tropism"]
    assert blocked == [{"task": "hvue_human_virus_pathogenicity_cini"}]
    assert merged_manifest == "merged.csv"


def test_checkpoint_specs_adds_option_b_best_candidate(tmp_path: Path, monkeypatch) -> None:
    best = tmp_path / "best_candidate.json"
    option_b_weights = tmp_path / "option_b.safetensors"
    option_b_weights.write_text("placeholder")
    best.write_text(
        json.dumps(
            {
                "best_candidate": {
                    "config_id": "retain_heavy_40x40",
                    "weights_path": str(option_b_weights),
                }
            }
        )
    )
    monkeypatch.setattr(
        "phase2.run_stage1_formal_experiment.FORMAL_CHECKPOINTS",
        {"projection_rank32": str(tmp_path / "projection" / "weights.safetensors")},
    )
    (tmp_path / "projection").mkdir()
    (tmp_path / "projection" / "weights.safetensors").write_text("placeholder")
    monkeypatch.setattr("phase2.run_stage1_formal_experiment.PROJECT_ROOT", Path("/"))

    specs, missing = checkpoint_specs(best)

    assert [row["name"] for row in specs] == ["base", "projection_rank32", "option_b_retain_heavy_40x40"]
    assert missing == []


def test_build_commands_cover_probe_and_lora_grid(tmp_path: Path) -> None:
    args = argparse.Namespace(
        python_bin="python",
        out_root=str(tmp_path / "out"),
        checkpoint_mode="base_only",
        include_probe_vs_sft=False,
        kmer_baseline_csv="kmer.csv",
        device="cuda:0",
        layers="3-9",
        sft_layer=9,
        seeds=[42, 43],
        classification_head_lrs=[1e-5, 5e-5],
        lora_lrs=[1e-5, 1e-4],
        lora_ranks=[8, 16],
        max_length=512,
        cpu_threads=16,
        probe_jobs=7,
        feature_batch_size=0,
        auto_batch_size=64,
        classification_head_batch_size=2,
        sft_steps=500,
        sft_eval_every=50,
        sft_patience=5,
        train_batch_size=1,
        eval_batch_size=1,
        lora_epochs=3,
        lora_max_steps=0,
        lora_eval_every=200,
        validation_max_rows=0,
        test_max_rows=0,
        lora_dropout=0.0,
        metric_for_best="auroc",
        split_type="cluster_disjoint",
    )
    tasks = ["hvue_human_host_tropism"]
    specs = [
        {"name": "base", "path": ""},
        {"name": "projection_rank32", "path": "projection.safetensors"},
        {"name": "option_b_retain_heavy_40x40", "path": "option_b.safetensors"},
    ]
    baseline_map = {("hvue_human_host_tropism", "cluster_disjoint"): 0.77}

    probe_commands = build_probe_vs_sft_commands(args, merged_manifest="merged.csv", tasks=tasks, specs=specs)
    lora_commands = build_lora_commands(
        args,
        merged_manifest="merged.csv",
        tasks=tasks,
        specs=specs,
        baseline_map=baseline_map,
    )

    assert len(probe_commands) == 2
    assert probe_commands[0]["family"] == "probe_vs_sft"
    assert "--checkpoints" in probe_commands[0]["cmd"]
    checkpoint_arg = probe_commands[0]["cmd"][probe_commands[0]["cmd"].index("--checkpoints") + 1]
    assert "projection_rank32=projection.safetensors" in checkpoint_arg
    assert len(lora_commands) == 3 * 2 * 2 * 2
    assert lora_commands[0]["family"] == "fresh_lora"
    assert "--kmer-baseline-score" in lora_commands[0]["cmd"]


def test_checkpoint_specs_can_skip_modified_dependencies(tmp_path: Path) -> None:
    best = tmp_path / "missing_best_candidate.json"

    specs, missing = checkpoint_specs(best, include_modified=False)

    assert specs == [{"name": "base", "path": ""}]
    assert missing == []


def test_load_kmer_baseline_map_reads_auroc(tmp_path: Path) -> None:
    baseline = tmp_path / "kmer.csv"
    baseline.write_text(
        "task,split_type,auroc\n"
        "hvue_human_host_tropism,cluster_disjoint,0.893\n"
    )

    payload = load_kmer_baseline_map(baseline)

    assert payload == {("hvue_human_host_tropism", "cluster_disjoint"): 0.893}
