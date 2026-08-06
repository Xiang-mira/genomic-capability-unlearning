import csv
import math
from pathlib import Path

import numpy as np

from phase2.stage1_baseline_alignment import (
    STRONG_MATCHED_PROTOCOL,
    aggregate_lora_runs,
    evaluate_kmer_protocol,
    lora_retained_region,
    read_manifest_samples,
    select_mcc_threshold,
)


def write_manifest(path: Path) -> None:
    rows = [
        ("train", "s1", "A" * 520, 0),
        ("train", "s2", "C" * 520, 0),
        ("train", "s3", "G" * 520, 1),
        ("train", "s4", "T" * 520, 1),
        ("val", "s5", "A" * 520, 0),
        ("val", "s6", "C" * 520, 0),
        ("val", "s7", "G" * 520, 1),
        ("val", "s8", "T" * 520, 1),
        ("test", "s9", "A" * 520, 0),
        ("test", "s10", "C" * 520, 0),
        ("test", "s11", "G" * 520, 1),
        ("test", "s12", "T" * 520, 1),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["benchmark", "task", "split", "split_type", "sequence", "label", "family", "group", "id"],
        )
        writer.writeheader()
        for split, sample_id, seq, label in rows:
            writer.writerow(
                {
                    "benchmark": "hvue",
                    "task": "hvue_human_host_tropism",
                    "split": split,
                    "split_type": "cluster_disjoint",
                    "sequence": seq,
                    "label": label,
                    "family": "",
                    "group": "hvue_forget",
                    "id": sample_id,
                }
            )


def test_lora_retained_region_is_deterministic_prefix() -> None:
    start, end, retained = lora_retained_region("ACGT" * 200)
    assert start == 0
    assert end == 512
    assert retained == ("ACGT" * 200)[:512]


def test_read_manifest_preserves_same_samples_and_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest)
    samples = read_manifest_samples(manifest, "hvue_human_host_tropism", "cluster_disjoint")
    assert [sample.sample_id for sample in samples[:3]] == ["s1", "s2", "s3"]
    assert {sample.split for sample in samples} == {"train", "val", "test"}
    assert sum(sample.split == "train" for sample in samples) == 4


def test_strong_matched_c_grid_includes_100() -> None:
    assert 100.0 in STRONG_MATCHED_PROTOCOL.c_grid
    assert STRONG_MATCHED_PROTOCOL.c_grid[-1] == 100.0


def test_validation_only_mcc_threshold_selection_ignores_test_labels() -> None:
    val_y = np.array([0, 0, 1, 1])
    val_p = np.array([0.1, 0.2, 0.7, 0.8])
    threshold_a, _ = select_mcc_threshold(val_y, val_p)
    test_y = np.array([1, 1, 0, 0])
    threshold_b, _ = select_mcc_threshold(val_y, val_p)
    assert threshold_a == threshold_b
    assert list(test_y) == [1, 1, 0, 0]


def test_evaluate_kmer_exports_predictions_and_uses_val_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest)
    samples = read_manifest_samples(manifest, "hvue_human_host_tropism", "cluster_disjoint")
    metrics = evaluate_kmer_protocol(samples, STRONG_MATCHED_PROTOCOL, tmp_path / "pred", seed=7)
    assert metrics["selection_split"] == "val"
    assert metrics["selected_c"] in STRONG_MATCHED_PROTOCOL.c_grid
    assert Path(metrics["prediction_paths"]["val"]).exists()
    assert Path(metrics["prediction_paths"]["test"]).exists()
    with Path(metrics["prediction_paths"]["test"]).open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert {"sample_id", "split", "label", "probability_positive", "selected_threshold"} <= set(row)
    assert {candidate["C"] for candidate in metrics["c_candidates"]} == set(STRONG_MATCHED_PROTOCOL.c_grid)


def test_auroc_excess_arithmetic_and_missing_lora_mcc_status() -> None:
    runs = [{"run_id": "r", "rank": 8, "lr": 1e-5, "seed": 42, "raw_lora_auroc": 0.9, "raw_lora_mcc": 0.5}]
    earlier = {"test_metrics_validation_threshold": {"auroc": 0.8, "mcc": 0.3}}
    strong = {"test_metrics_validation_threshold": {"auroc": 0.85, "mcc": 0.4}}
    full = {"test_metrics_validation_threshold": {"auroc": 0.88, "mcc": 0.45}}
    enriched, summary, judgement = aggregate_lora_runs(runs, earlier, strong, full)
    assert math.isclose(enriched[0]["excess_vs_earlier_matched_input_kmer"], 0.1)
    assert math.isclose(enriched[0]["excess_vs_strong_matched_input_kmer"], 0.05)
    assert enriched[0]["mcc_status"] == "missing_lora_validation_and_test_prediction_exports"
    assert summary[0]["formal_mcc_status"] == "missing_lora_prediction_exports"
    assert judgement["selected_formal_attacker"] is None


def test_kmer_reproducible_with_same_seed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest)
    samples = read_manifest_samples(manifest, "hvue_human_host_tropism", "cluster_disjoint")
    first = evaluate_kmer_protocol(samples, STRONG_MATCHED_PROTOCOL, tmp_path / "pred1", seed=11)
    second = evaluate_kmer_protocol(samples, STRONG_MATCHED_PROTOCOL, tmp_path / "pred2", seed=11)
    assert first["selected_c"] == second["selected_c"]
    assert first["test_metrics_validation_threshold"]["auroc"] == second["test_metrics_validation_threshold"]["auroc"]


def test_confirmation_seed_does_not_replace_development_selection() -> None:
    runs = [
        {"run_id": "dev42", "rank": 8, "lr": 1e-5, "seed": 42, "raw_lora_auroc": 0.84, "raw_lora_mcc": 0.1},
        {"run_id": "dev43", "rank": 8, "lr": 1e-5, "seed": 43, "raw_lora_auroc": 0.84, "raw_lora_mcc": 0.1},
        {"run_id": "confirm44", "rank": 8, "lr": 1e-5, "seed": 44, "raw_lora_auroc": 0.99, "raw_lora_mcc": 0.9},
    ]
    baseline = {"test_metrics_validation_threshold": {"auroc": 0.85, "mcc": 0.0}}
    _enriched, summary, judgement = aggregate_lora_runs(runs, baseline, baseline, baseline)
    assert summary[0]["dev_seed_auroc_excess_vs_strong_matched_all_positive"] is False
    assert judgement["best_development_configuration"] is None


def test_formal_grid_aggregation_keeps_only_lora_runs_provided() -> None:
    runs = [{"run_id": "formal", "rank": 8, "lr": 1e-5, "seed": 42, "raw_lora_auroc": 0.9, "raw_lora_mcc": 0.1}]
    baseline = {"test_metrics_validation_threshold": {"auroc": 0.8, "mcc": 0.0}}
    enriched, summary, _judgement = aggregate_lora_runs(runs, baseline, baseline, baseline)
    assert [row["run_id"] for row in enriched] == ["formal"]
    assert len(summary) == 1
