from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.build_stage2_attacked_checkpoints import build_commands, update_variant_spec, write_stage2_run_metadata


def make_args(**overrides):
    defaults = dict(
        python_bin="python",
        variant_spec_json="variants.json",
        benchmark_manifest="formal.csv",
        target_task="hvue_human_host_tropism",
        recipes="lora_r8_lr1e5_l5l9,full_lr1e5_all",
        split_type="cluster_disjoint",
        device="cpu",
        cpu_threads=1,
        train_batch_size=1,
        eval_batch_size=1,
        max_length=128,
        epochs=1,
        max_steps=10,
        eval_every=5,
        validation_max_rows=64,
        test_max_rows=64,
        lora_alpha=16,
        lora_dropout=0.0,
        metric_for_best="auto",
        export_attack_policy="delta",
        export_attack_suffixes="all",
        seed=42,
        out_dir="out",
        execute=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_commands_marks_lora_ready_and_full_ft_blocked(tmp_path: Path) -> None:
    spec_path = tmp_path / "variants.json"
    spec_path.write_text(
        json.dumps(
            [
                {
                    "variant_id": "option_a_base",
                    "initializer_label": "none",
                    "k0_ckpt": "",
                    "attacked_ckpt_by_recipe": {},
                    "recipe_ids": ["k0_no_attack", "lora_r8_lr1e5_l5l9", "full_lr1e5_all"],
                },
                {
                    "variant_id": "option_b_classification_ce",
                    "initializer_label": "classification_ce",
                    "k0_ckpt": "best_option_b.safetensors",
                    "attacked_ckpt_by_recipe": {},
                    "recipe_ids": ["k0_no_attack", "lora_r8_lr1e5_l5l9"],
                    "readout_disruption_flag": "readout_disruption",
                },
            ]
        )
    )
    args = make_args(variant_spec_json=str(spec_path), out_dir=str(tmp_path / "out"))

    commands = build_commands(args)

    ready = [row for row in commands if row["status"] == "ready"]
    blocked = [row for row in commands if row["status"] == "blocked_unsupported_attack_method"]
    assert len(ready) == 2
    assert len(blocked) == 2
    assert ready[0]["recipe_id"] == "lora_r8_lr1e5_l5l9"
    assert "--export-attack-ckpt-dir" in ready[0]["command"]
    assert "--ckpt" in ready[1]["command"]
    assert {row["recipe_id"] for row in blocked} == {"full_lr1e5_all"}


def test_update_variant_spec_wires_expected_attack_paths(tmp_path: Path) -> None:
    spec_path = tmp_path / "variants.json"
    spec_path.write_text(
        json.dumps(
            [
                {"variant_id": "option_a_base", "attacked_ckpt_by_recipe": {}},
                {"variant_id": "option_b_classification_ce", "attacked_ckpt_by_recipe": {}},
            ]
        )
    )
    out_path = tmp_path / "updated.json"
    update_variant_spec(
        str(spec_path),
        [
            {
                "variant_id": "option_b_classification_ce",
                "recipe_id": "lora_r8_lr1e5_l5l9",
                "status": "ready",
                "expected_exported_weights": "out/option_b/lora/weights.safetensors",
            }
        ],
        out_path,
    )

    updated = json.loads(out_path.read_text())
    option_b = next(row for row in updated if row["variant_id"] == "option_b_classification_ce")
    assert option_b["attacked_ckpt_by_recipe"]["lora_r8_lr1e5_l5l9"] == "out/option_b/lora/weights.safetensors"
    assert option_b["recipe_ids"] == ["lora_r8_lr1e5_l5l9"]


def test_write_stage2_run_metadata_summarizes_commands(tmp_path: Path) -> None:
    manifest = tmp_path / "formal.csv"
    manifest.write_text("task\n")
    variants = tmp_path / "variants.json"
    variants.write_text("[]\n")
    args = make_args(
        variant_spec_json=str(variants),
        benchmark_manifest=str(manifest),
        out_dir=str(tmp_path / "out"),
        recipes="lora_r8_lr1e5_l5l9,full_lr1e5_all",
    )
    commands = [
        {"variant_id": "option_a", "recipe_id": "lora_r8_lr1e5_l5l9", "status": "ready"},
        {"variant_id": "option_a", "recipe_id": "full_lr1e5_all", "status": "blocked_unsupported_attack_method"},
    ]

    metadata_path = write_stage2_run_metadata(args, commands, tmp_path)

    payload = json.loads(metadata_path.read_text())
    assert payload["phase"] == "stage2_attacked_checkpoint_build"
    assert payload["ready_count"] == 1
    assert payload["blocked_count"] == 1
