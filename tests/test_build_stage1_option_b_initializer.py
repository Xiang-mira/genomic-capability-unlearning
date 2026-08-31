from __future__ import annotations

import argparse
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
phase1_utils_module.read_manifest = lambda *args, **kwargs: []
phase1_module.utils = phase1_utils_module
# If this stub is actually installed (bare environment, no torch/stripedhyena),
# keep the real package search path on it so sibling modules such as
# phase1.build_refseq_family_target_dataset still import from disk instead of
# failing with "'phase1' is not a package".
phase1_module.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "phase1")]
register_stub("phase1", phase1_module)
register_stub("phase1.utils", phase1_utils_module)

checkpoint_io_module = types.ModuleType("phase2.checkpoint_io")
checkpoint_io_module.save_checkpoint = lambda *args, **kwargs: None
checkpoint_io_module.set_trainable_by_suffixes = lambda *args, **kwargs: []
checkpoint_io_module.snapshot_state = lambda *args, **kwargs: {}
register_stub("phase2.checkpoint_io", checkpoint_io_module)

eval_benchmarks_module = types.ModuleType("phase2.eval_benchmarks")

class _BenchmarkRecord:
    def __init__(self, split: str, label: str, record_id: str):
        self.split = split
        self.label = label
        self.record_id = record_id
        self.sequence = record_id

eval_benchmarks_module.BenchmarkRecord = _BenchmarkRecord
eval_benchmarks_module.apply_checkpoint = lambda *args, **kwargs: None
eval_benchmarks_module.read_benchmark_manifest = lambda *args, **kwargs: []
register_stub("phase2.eval_benchmarks", eval_benchmarks_module)

lora_utils_module = types.ModuleType("phase2.lora_utils")
lora_utils_module.PooledEvoClassifier = object
lora_utils_module.encode_labels = lambda labels: (None, types.SimpleNamespace(label_to_id={"0": 0, "1": 1}, num_classes=2))
register_stub("phase2.lora_utils", lora_utils_module)

phase2_utils_module = types.ModuleType("phase2.utils")
phase2_utils_module.get_trainable_params = lambda *args, **kwargs: []
phase2_utils_module.iterate_batches = lambda records, batch_size, shuffle, rng: iter([records[:batch_size]])
phase2_utils_module.language_model_loss = lambda *args, **kwargs: None
phase2_utils_module.tokenize_batch = lambda *args, **kwargs: None
register_stub("phase2.utils", phase2_utils_module)

from phase2.build_stage1_option_b_initializer import (
    build_initializer_metadata,
    deterministic_label_subset,
    split_records,
    summarize_labels,
)

for module_name in [
    "phase1",
    "phase1.utils",
    "phase2.checkpoint_io",
    "phase2.eval_benchmarks",
    "phase2.lora_utils",
    "phase2.utils",
    "evo",
    "evo.tokenizer",
]:
    sys.modules.pop(module_name, None)


def test_split_records_normalizes_dev_to_val() -> None:
    records = [
        _BenchmarkRecord("train", "0", "a"),
        _BenchmarkRecord("dev", "1", "b"),
        _BenchmarkRecord("test", "0", "c"),
    ]
    splits = split_records(records)
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1


def test_deterministic_label_subset_preserves_label_coverage() -> None:
    records = [
        _BenchmarkRecord("train", "0", "a"),
        _BenchmarkRecord("train", "0", "b"),
        _BenchmarkRecord("train", "1", "c"),
        _BenchmarkRecord("train", "1", "d"),
    ]
    subset = deterministic_label_subset(records, 2, 42, "train")
    assert len(subset) == 2
    assert sorted(record.label for record in subset) == ["0", "1"]


def test_summarize_labels_counts_classes() -> None:
    records = [_BenchmarkRecord("train", "0", "a"), _BenchmarkRecord("train", "1", "b"), _BenchmarkRecord("train", "1", "c")]
    assert summarize_labels(records) == {"0": 1, "1": 2}


def test_build_initializer_metadata_includes_run_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "formal.csv"
    retain = tmp_path / "retain.csv"
    manifest.write_text("task\n")
    retain.write_text("sequence\n")
    args = argparse.Namespace(
        target_task="hvue_human_host_tropism",
        split_type="cluster_disjoint",
        benchmark_manifest=str(manifest),
        retain_csv=str(retain),
        trainable_suffixes="all",
        checkpoint_policy="delta",
        elicitation_steps=20,
        ascent_steps=20,
        alpha_target=1.0,
        alpha_retain=1.0,
        model_dir="./evo-1-8k-base",
        init_ckpt="",
        seed=42,
    )
    weights_path = tmp_path / "weights.safetensors"
    save_result = types.SimpleNamespace(saved=True, tensor_count=7, checkpoint_policy="delta")
    target_splits = {
        "train": [_BenchmarkRecord("train", "0", "a"), _BenchmarkRecord("train", "1", "b")],
        "val": [_BenchmarkRecord("val", "0", "c"), _BenchmarkRecord("val", "1", "d")],
        "test": [_BenchmarkRecord("test", "0", "e"), _BenchmarkRecord("test", "1", "f")],
    }

    payload = build_initializer_metadata(
        args,
        weights_path=weights_path,
        save_result=save_result,
        layers=[5, 6, 7, 8, 9],
        selected_names={"blocks.5.mlp.weight", "blocks.6.mlp.weight"},
        target_splits=target_splits,
        retain_records=[types.SimpleNamespace(record_id="r1"), types.SimpleNamespace(record_id="r2")],
        final_val_metrics={"auroc": 0.55},
        final_test_metrics={"auroc": 0.51},
    )

    assert payload["method"] == "tar_option_b_initializer"
    assert payload["commit_hash"] is not None
    assert payload["config_hash"] is not None
    assert payload["source_checkpoint"] == "./evo-1-8k-base"
    assert payload["output_checkpoint"] == str(weights_path)
    assert payload["target_counts"]["train"] == 2
    assert payload["readout_disruption_flag"] == "readout_disruption"
