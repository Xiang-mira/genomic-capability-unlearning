import csv
import json
from pathlib import Path

from phase2.lora_subspace_targeting import (
    build_random_label_manifest,
    build_rerun_plan,
    inventory_stage1_runs,
    run,
)


def write_stage1_run(root: Path, rank: int, lr_label: str, seed: int, checkpoint_retained: bool = False) -> None:
    run_dir = root / f"fresh_lora/base/rank_{rank}/lr_{lr_label}/seed_{seed}"
    run_dir.mkdir(parents=True)
    with (run_dir / "eval_benchmarks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "auroc",
                "mcc",
                "lora_alpha",
                "checkpoint_retained",
                "best_checkpoint",
                "validation_prediction_path",
                "test_prediction_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "auroc": "0.9",
                "mcc": "0.5",
                "lora_alpha": str(rank * 2),
                "checkpoint_retained": str(checkpoint_retained),
                "best_checkpoint": "",
                "validation_prediction_path": "",
                "test_prediction_path": "",
            }
        )
    (run_dir / "eval_benchmarks_metadata.json").write_text(
        json.dumps(
            {
                "data_hashes": {
                    "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv": "manifesthash"
                }
            }
        )
    )


def write_alignment_report(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "stage1_baseline_alignment_report.json").write_text(
        json.dumps(
            {
                "conclusion": {
                    "stable_positive_auroc_configurations": [
                        {"rank": 32, "lr": 1e-5, "auroc_std": 0.001},
                        {"rank": 16, "lr": 5e-5, "auroc_std": 0.01},
                    ]
                }
            }
        )
    )


def write_formal_manifest(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["benchmark", "task", "split", "split_type", "sequence", "label", "family", "group", "id"],
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            for idx, label in enumerate(["0", "0", "1", "1"]):
                writer.writerow(
                    {
                        "benchmark": "hvue",
                        "task": "hvue_human_host_tropism",
                        "split": split,
                        "split_type": "cluster_disjoint",
                        "sequence": "ACGT" * 200,
                        "label": label,
                        "family": "",
                        "group": "hvue_forget",
                        "id": f"{split}-{idx}",
                    }
                )


def test_inventory_marks_discarded_checkpoints_and_missing_predictions(tmp_path: Path) -> None:
    write_stage1_run(tmp_path, 32, "5e-5", 44)
    rows = inventory_stage1_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["checkpoint_discard_status"] == "discarded"
    assert rows[0]["lora_a_matrices_present"] is False
    assert rows[0]["validation_probabilities_present"] is False
    assert rows[0]["manifest_hash"] == "manifesthash"


def test_rerun_plan_deduplicates_stable_best_and_exploratory_configs(tmp_path: Path) -> None:
    alignment = tmp_path / "alignment"
    write_alignment_report(alignment)
    inventory = [{"adapter_path": "", "validation_prediction_path": "", "test_prediction_path": ""}]
    random_manifest = tmp_path / "random.csv"
    plan = build_rerun_plan(inventory, alignment, tmp_path / "out", random_label_manifest=random_manifest)
    normal = [row for row in plan["planned_reruns"] if row["status"] == "planned_not_started"]
    configs = {(row["rank"], row["learning_rate"]) for row in normal}
    assert (32, 1e-5) in configs
    assert (32, 5e-5) in configs
    assert (16, 5e-5) in configs
    assert (16, 1e-5) in configs
    assert any("--export-predictions" in row["command"] for row in normal)
    assert "--discard-task-checkpoint" not in " ".join(" ".join(row["command"]) for row in normal)
    controls = [row for row in normal if row["selection_label"] == "matched_random_label_control"]
    assert len(controls) == 3
    assert all(str(random_manifest) in row["command"] for row in controls)


def test_random_label_manifest_preserves_ids_splits_sequences_and_label_counts(tmp_path: Path) -> None:
    source = tmp_path / "formal.csv"
    target = tmp_path / "random.csv"
    write_formal_manifest(source)
    meta = build_random_label_manifest(source, target, seed=123)
    original = read_rows(source)
    randomized = read_rows(target)
    assert [row["id"] for row in original] == [row["id"] for row in randomized]
    assert [row["split"] for row in original] == [row["split"] for row in randomized]
    assert [row["sequence"] for row in original] == [row["sequence"] for row in randomized]
    for split in ("train", "val", "test"):
        old = sorted(row["label"] for row in original if row["split"] == split)
        new = sorted(row["label"] for row in randomized if row["split"] == split)
        assert old == new
    assert meta["random_label_manifest_sha256"]


def test_run_writes_task1_deliverables_and_gated_report(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1"
    alignment = tmp_path / "alignment"
    out = tmp_path / "out"
    write_stage1_run(stage1, 32, "5e-5", 44)
    write_alignment_report(alignment)

    args = type("Args", (), {})()
    args.stage1_root = stage1
    args.alignment_root = alignment
    args.manifest = tmp_path / "formal.csv"
    args.random_label_seed = 1042
    args.out_dir = out
    write_formal_manifest(args.manifest)
    run(args)
    assert (out / "adapter_inventory.csv").exists()
    assert (out / "missing_artifacts_rerun_plan.json").exists()
    assert (out / "manifests" / "hvue_human_host_tropism_random_labels_seed1042.csv").exists()
    report = json.loads((out / "final_lora_subspace_targeting_report.json").read_text())
    assert report["status"] == "blocked_at_task2_targeted_adapter_reruns_not_started"
    assert report["go_no_go"] == "not_reached"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
