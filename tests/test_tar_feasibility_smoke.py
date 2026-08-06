from __future__ import annotations

import csv
import json
from pathlib import Path

from phase2.tar_feasibility_smoke import (
    VariantSpec,
    backfill_fresh_head_score,
    backfill_kmer_baseline_scores,
    build_command,
    load_kmer_baseline_map,
    load_variant_specs,
    parse_recipes,
    parse_tasks,
    apply_backfills,
    summarize_results,
    validate_requested_split_type,
    write_smoke_run_metadata,
)


def test_parse_tasks_defaults_to_formal_targets() -> None:
    tasks = parse_tasks("")
    assert "hvue_human_host_tropism" in tasks
    assert "hvue_human_virus_pathogenicity_cini" in tasks


def test_parse_recipes_empty_means_all() -> None:
    assert parse_recipes("") is None


def test_parse_recipes_preserves_requested_order() -> None:
    assert parse_recipes("k0_no_attack,full_lr1e5_all") == ["k0_no_attack", "full_lr1e5_all"]


def test_build_command_uses_attack_checkpoint_only_for_attacked_recipe(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(
        benchmark_manifest="manifest.csv",
        attacked_ckpt="weights.safetensors",
        device="cpu",
        cpu_threads=1,
        train_batch_size=1,
        eval_batch_size=1,
        max_length=32,
        epochs=1,
        max_steps=1,
        eval_every=1,
        validation_max_rows=8,
        test_max_rows=16,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        seed=42,
        split_type="cluster_disjoint",
    )
    variant = VariantSpec(variant_id="option_a", attacked_ckpt="weights.safetensors")
    cmd_k0 = build_command("python", tmp_path, args, variant, "k0_no_attack", "task_a", tmp_path / "k0")
    cmd_attack = build_command("python", tmp_path, args, variant, "lora_r8_lr1e5_l5l9", "task_a", tmp_path / "attack")

    assert "--ckpt" not in cmd_k0
    assert "--ckpt" in cmd_attack


def test_build_command_can_attach_k0_initializer_checkpoint(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(
        benchmark_manifest="manifest.csv",
        attacked_ckpt="",
        device="cpu",
        cpu_threads=1,
        train_batch_size=1,
        eval_batch_size=1,
        max_length=32,
        epochs=1,
        max_steps=1,
        eval_every=1,
        validation_max_rows=8,
        test_max_rows=16,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        seed=42,
        split_type="cluster_disjoint",
    )
    variant = VariantSpec(
        variant_id="option_b",
        attacked_ckpt_by_recipe={"lora_r8_lr1e5_l5l9": "attack.safetensors"},
        k0_ckpt="initializer.safetensors",
        readout_disruption_flag="readout_disruption",
    )
    cmd_k0 = build_command("python", tmp_path, args, variant, "k0_no_attack", "task_a", tmp_path / "k0")
    cmd_attack = build_command("python", tmp_path, args, variant, "lora_r8_lr1e5_l5l9", "task_a", tmp_path / "attack")

    assert cmd_k0[:5] == ["python", "-u", str(tmp_path / "phase2" / "eval_benchmarks.py"), "--ckpt", "initializer.safetensors"]
    assert "--readout-disruption-flag" in cmd_k0
    assert cmd_attack[:5] == ["python", "-u", str(tmp_path / "phase2" / "eval_benchmarks.py"), "--ckpt", "attack.safetensors"]


def test_load_variant_specs_reads_json_variants(tmp_path: Path) -> None:
    import argparse

    spec_path = tmp_path / "variants.json"
    spec_path.write_text(
        '[{"variant_id":"Option B init","k0_ckpt":"init.safetensors","attacked_ckpt_by_recipe":{"full_lr1e5_all":"full.safetensors"},"initializer_label":"classification_ce","readout_disruption_flag":"readout_disruption","recipe_ids":["k0_no_attack"]}]'
    )
    args = argparse.Namespace(attacked_ckpt="", variant_spec_json=str(spec_path), variant_id="")

    variants = load_variant_specs(args)

    assert len(variants) == 1
    assert variants[0].variant_id == "option_b_init"
    assert variants[0].k0_ckpt == "init.safetensors"
    assert variants[0].attacked_ckpt_by_recipe == {"full_lr1e5_all": "full.safetensors"}
    assert variants[0].initializer_label == "classification_ce"
    assert variants[0].recipe_ids == ("k0_no_attack",)


def test_variant_recipe_ids_can_limit_smoke_commands(tmp_path: Path) -> None:
    import argparse

    spec_path = tmp_path / "variants.json"
    spec_path.write_text(
        '[{"variant_id":"option_b","k0_ckpt":"init.safetensors","initializer_label":"classification_ce","recipe_ids":["k0_no_attack"]}]'
    )
    args = argparse.Namespace(attacked_ckpt="", variant_spec_json=str(spec_path), variant_id="")

    variants = load_variant_specs(args)

    assert variants[0].recipe_ids == ("k0_no_attack",)


def test_backfill_fresh_head_score_uses_primary_metric(tmp_path: Path) -> None:
    path = tmp_path / "eval_benchmarks.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "auroc", "mcc", "post_attack_fresh_head_score"],
        )
        writer.writeheader()
        writer.writerow({"task": "task_a", "auroc": "0.73", "mcc": "0.20", "post_attack_fresh_head_score": ""})

    backfill_fresh_head_score(path)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["post_attack_fresh_head_score"] == "0.73"


def test_load_kmer_baseline_map_uses_task_and_split_type(tmp_path: Path) -> None:
    path = tmp_path / "kmer.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split_type", "auroc", "mcc"])
        writer.writeheader()
        writer.writerow({"task": "task_a", "split_type": "cluster_disjoint", "auroc": "0.61", "mcc": "0.1"})
        writer.writerow({"task": "task_a", "split_type": "random", "auroc": "0.81", "mcc": "0.2"})

    baseline_map = load_kmer_baseline_map(str(path))

    assert baseline_map == {("task_a", "cluster_disjoint"): 0.61, ("task_a", "random"): 0.81}


def test_backfill_kmer_baseline_scores_populates_excess(tmp_path: Path) -> None:
    path = tmp_path / "eval_benchmarks.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "split_type", "auroc", "mcc", "kmer_baseline_score", "metric_excess_over_kmer"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "task_a",
                "split_type": "cluster_disjoint",
                "auroc": "0.73",
                "mcc": "0.20",
                "kmer_baseline_score": "",
                "metric_excess_over_kmer": "",
            }
        )

    backfill_kmer_baseline_scores(path, {("task_a", "cluster_disjoint"): 0.61})

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["kmer_baseline_score"] == "0.61"
    assert rows[0]["metric_excess_over_kmer"] == "0.12"


def test_summarize_results_collects_variant_recipe_rows(tmp_path: Path) -> None:
    result_dir = tmp_path / "option_a" / "k0_no_attack"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "eval_benchmarks.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "checkpoint",
                "split_type",
                "kmer_baseline_score",
                "metric_excess_over_kmer",
                "auroc",
                "mcc",
                "accuracy",
                "post_attack_fresh_head_score",
                "readout_disruption_flag",
                "n_test_eval",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "hvue_human_host_tropism",
                "checkpoint": "base",
                "split_type": "cluster_disjoint",
                "kmer_baseline_score": "0.52",
                "metric_excess_over_kmer": "0.08",
                "auroc": "0.6",
                "mcc": "0.1",
                "accuracy": "0.5",
                "post_attack_fresh_head_score": "0.6",
                "readout_disruption_flag": "",
                "n_test_eval": "64",
            }
        )

    summary_path = summarize_results(
        [
            {
                "variant_id": "option_a",
                "initializer_label": "none",
                "recipe_id": "k0_no_attack",
                "out_dir": str(result_dir),
            }
        ],
        tmp_path,
    )

    assert summary_path == tmp_path / "stage1_smoke_summary.csv"
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["variant_id"] == "option_a"
    assert rows[0]["recipe_id"] == "k0_no_attack"
    assert rows[0]["task"] == "hvue_human_host_tropism"
    assert rows[0]["kmer_baseline_score"] == "0.52"


def test_apply_backfills_updates_existing_result_files(tmp_path: Path) -> None:
    result_dir = tmp_path / "option_a" / "k0_no_attack"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "eval_benchmarks.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "split_type",
                "auroc",
                "mcc",
                "post_attack_fresh_head_score",
                "kmer_baseline_score",
                "metric_excess_over_kmer",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "task_a",
                "split_type": "cluster_disjoint",
                "auroc": "0.73",
                "mcc": "0.2",
                "post_attack_fresh_head_score": "",
                "kmer_baseline_score": "",
                "metric_excess_over_kmer": "",
            }
        )

    apply_backfills(
        [{"out_dir": str(result_dir)}],
        {("task_a", "cluster_disjoint"): 0.61},
    )

    with result_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["post_attack_fresh_head_score"] == "0.73"
    assert rows[0]["kmer_baseline_score"] == "0.61"


def test_validate_requested_split_type_requires_matching_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split_type", "split", "sequence", "label"])
        writer.writeheader()
        writer.writerow({"task": "task_a", "split_type": "random", "split": "train", "sequence": "ACGT", "label": "0"})

    try:
        validate_requested_split_type(str(path), ["task_a"], "cluster_disjoint")
    except ValueError as exc:
        assert "does not contain split_type=cluster_disjoint rows" in str(exc)
    else:
        raise AssertionError("expected validate_requested_split_type to fail")


def test_validate_requested_split_type_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split", "sequence", "label"])
        writer.writeheader()
        writer.writerow({"task": "task_a", "split": "train", "sequence": "ACGT", "label": "0"})

    try:
        validate_requested_split_type(str(path), ["task_a"], "cluster_disjoint")
    except ValueError as exc:
        assert "does not contain a split_type column" in str(exc)
    else:
        raise AssertionError("expected validate_requested_split_type to fail")


def test_write_smoke_run_metadata_records_tasks_and_variants(tmp_path: Path) -> None:
    import argparse

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("task,split_type\n")
    baseline = tmp_path / "kmer.csv"
    baseline.write_text("task,split_type,auroc\n")
    variant_json = tmp_path / "variants.json"
    variant_json.write_text("[]\n")
    args = argparse.Namespace(
        benchmark_manifest=str(manifest),
        kmer_baseline_csv=str(baseline),
        variant_spec_json=str(variant_json),
        split_type="cluster_disjoint",
        execute=False,
        backfill_only=False,
        seed=42,
    )
    variants = [VariantSpec(variant_id="option_a", initializer_label="none")]
    commands = [{"variant_id": "option_a", "recipe_id": "k0_no_attack", "out_dir": str(tmp_path / "out" / "a")}]

    metadata_path = write_smoke_run_metadata(args, tmp_path, ["hvue_human_host_tropism"], variants, commands)

    payload = json.loads(metadata_path.read_text())
    assert payload["phase"] == "tar_feasibility_smoke"
    assert payload["target_tasks"] == ["hvue_human_host_tropism"]
    assert payload["variant_ids"] == ["option_a"]
