from __future__ import annotations

import csv
import json
import pathlib
import sys
import types

from tests._stub_support import register_stub
from pathlib import Path

evo_module = types.ModuleType("evo")
evo_tokenizer_module = types.ModuleType("evo.tokenizer")
evo_tokenizer_module.CharLevelTokenizer = object
evo_module.tokenizer = evo_tokenizer_module
register_stub("evo", evo_module)
register_stub("evo.tokenizer", evo_tokenizer_module)

phase1_module = types.ModuleType("phase1")
phase1_utils_module = types.ModuleType("phase1.utils")
phase1_utils_module.load_local_checkpoint = lambda *args, **kwargs: None
phase1_module.utils = phase1_utils_module
# If this stub is actually installed (bare environment, no torch/stripedhyena),
# keep the real package search path on it so sibling modules such as
# phase1.build_refseq_family_target_dataset still import from disk instead of
# failing with "'phase1' is not a package".
phase1_module.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "phase1")]
register_stub("phase1", phase1_module)
register_stub("phase1.utils", phase1_utils_module)

phase2_lora_utils = types.ModuleType("phase2.lora_utils")
phase2_lora_utils.LabelEncoding = object
phase2_lora_utils.PooledEvoClassifier = object
phase2_lora_utils.classification_metrics = lambda *args, **kwargs: {}
phase2_lora_utils.count_total = lambda *args, **kwargs: 0
phase2_lora_utils.count_trainable = lambda *args, **kwargs: 0
phase2_lora_utils.encode_labels = lambda *args, **kwargs: (None, None)
phase2_lora_utils.freeze_all = lambda *args, **kwargs: None
phase2_lora_utils.inject_lora_all_blocks = lambda *args, **kwargs: ([], [])
phase2_lora_utils.merge_lora_adapters = lambda *args, **kwargs: []
phase2_lora_utils.regression_metrics = lambda *args, **kwargs: {}
phase2_lora_utils.remove_lora_adapters = lambda *args, **kwargs: None
register_stub("phase2.lora_utils", phase2_lora_utils)

phase2_notify = types.ModuleType("phase2.notify")
phase2_notify.notify = lambda *args, **kwargs: None
register_stub("phase2.notify", phase2_notify)

phase2_checkpoint_io = types.ModuleType("phase2.checkpoint_io")
phase2_checkpoint_io.apply_checkpoint = lambda *args, **kwargs: None
phase2_checkpoint_io.save_checkpoint = lambda *args, **kwargs: None
phase2_checkpoint_io.snapshot_state = lambda *args, **kwargs: {}
register_stub("phase2.checkpoint_io", phase2_checkpoint_io)

phase2_utils = types.ModuleType("phase2.utils")
phase2_utils.tokenize_batch = lambda *args, **kwargs: None
register_stub("phase2.utils", phase2_utils)

from phase2.eval_benchmarks import (
    BenchmarkRecord,
    maybe_export_attack_checkpoint,
    parse_layers_spec,
    read_benchmark_manifest,
    write_eval_run_metadata,
)


def test_parse_layers_spec_supports_ranges() -> None:
    assert parse_layers_spec("5-7,9") == [5, 6, 7, 9]


def test_read_benchmark_manifest_filters_requested_split_type(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark", "task", "split", "split_type", "sequence", "label", "group"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "benchmark": "hvue",
                "task": "task_a",
                "split": "train",
                "split_type": "random",
                "sequence": "ACGT",
                "label": "0",
                "group": "hvue_forget",
            }
        )
        writer.writerow(
            {
                "benchmark": "hvue",
                "task": "task_a",
                "split": "train",
                "split_type": "cluster_disjoint",
                "sequence": "TGCA",
                "label": "1",
                "group": "hvue_forget",
            }
        )

    records = read_benchmark_manifest(
        str(path),
        benchmark_scope="task",
        task_filter={"task_a"},
        requested_split_type="cluster_disjoint",
    )

    assert len(records) == 1
    assert records[0].sequence == "TGCA"


def test_read_benchmark_manifest_rejects_requested_split_without_column(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark", "task", "split", "sequence", "label", "group"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "benchmark": "hvue",
                "task": "task_a",
                "split": "train",
                "sequence": "ACGT",
                "label": "0",
                "group": "hvue_forget",
            }
        )

    try:
        read_benchmark_manifest(
            str(path),
            benchmark_scope="task",
            task_filter={"task_a"},
            requested_split_type="cluster_disjoint",
        )
    except ValueError as exc:
        assert "does not contain a split_type column" in str(exc)
    else:
        raise AssertionError("expected read_benchmark_manifest to fail")


def test_write_eval_run_metadata_records_scope_and_tasks(tmp_path) -> None:
    import argparse

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("task\n")
    ckpt = tmp_path / "weights.safetensors"
    ckpt.write_text("placeholder\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = argparse.Namespace(
        benchmark_manifest=str(manifest),
        ckpt=str(ckpt),
        seed=42,
        benchmark_scope="task",
        task_filter="task_a,task_b",
        split_type="cluster_disjoint",
        export_attack_ckpt_dir="exports",
        export_attack_policy="delta",
        export_attack_layers="5-9",
        export_attack_suffixes="all",
        attack_recipe_id="lora_r8_lr1e5_l5l9",
    )

    write_eval_run_metadata(
        args=args,
        out_dir=str(out_dir),
        checkpoint_label="weights.safetensors",
        task_items=[("task_a", [BenchmarkRecord("hvue", "task_a", "train", "ACGT", "0")])],
        rows=[{"task": "task_a"}],
        results_path=str(out_dir / "eval_benchmarks.csv"),
        summary_path=str(out_dir / "eval_benchmarks_summary.json"),
        progress_path=str(out_dir / "eval_benchmarks_progress.json"),
    )

    payload = json.loads((out_dir / "eval_benchmarks_metadata.json").read_text())
    assert payload["phase"] == "eval_benchmarks"
    assert payload["checkpoint_label"] == "weights.safetensors"
    assert payload["task_count"] == 1
    assert payload["completed_tasks"] == ["task_a"]
    assert payload["attack_recipe_id"] == "lora_r8_lr1e5_l5l9"


def test_maybe_export_attack_checkpoint_writes_provenance_meta(tmp_path, monkeypatch) -> None:
    import argparse

    saved = {}

    def fake_save_checkpoint(model, path, policy, layers, suffixes, init_state, metadata):
        saved["path"] = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("weights\n")
        saved["metadata"] = metadata
        return types.SimpleNamespace(saved=True, tensor_count=3, checkpoint_policy=policy)

    monkeypatch.setattr("phase2.eval_benchmarks.save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr("phase2.eval_benchmarks.merge_lora_adapters", lambda model: ["blocks.5.attn"])

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("task\n")
    ckpt = tmp_path / "init.safetensors"
    ckpt.write_text("init\n")
    export_root = tmp_path / "exports"
    args = argparse.Namespace(
        export_attack_ckpt_dir=str(export_root),
        export_attack_policy="delta",
        export_attack_layers="5-9",
        export_attack_suffixes="all",
        attack_recipe_id="lora_r8_lr1e5_l5l9",
        benchmark_manifest=str(manifest),
        ckpt=str(ckpt),
        seed=42,
    )

    output = maybe_export_attack_checkpoint(
        args=args,
        model=object(),
        task="hvue_human_host_tropism",
        checkpoint_label="base",
        best_payload={"step": 10, "metric_for_best": "auroc", "selection_value": 0.6},
        init_state={},
    )

    meta = json.loads((export_root / "hvue_human_host_tropism" / "meta.json").read_text())
    assert output.endswith("weights.safetensors")
    assert meta["phase"] == "eval_benchmarks_attack_export"
    assert meta["task"] == "hvue_human_host_tropism"
    assert meta["attack_recipe_id"] == "lora_r8_lr1e5_l5l9"
    assert meta["output_checkpoint"] == output
