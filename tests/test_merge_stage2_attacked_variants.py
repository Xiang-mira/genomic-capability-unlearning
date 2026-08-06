from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from phase2.merge_stage2_attacked_variants import merge_variant_rows


def test_merge_variant_rows_combines_recipe_maps_and_recipe_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            [
                {
                    "variant_id": "option_a_base",
                    "initializer_label": "none",
                    "recipe_ids": ["k0_no_attack", "lora_r8_lr1e5_l5l9"],
                    "attacked_ckpt_by_recipe": {"lora_r8_lr1e5_l5l9": "a_r8.safetensors"},
                },
                {
                    "variant_id": "option_b_classification_ce",
                    "initializer_label": "classification_ce",
                    "k0_ckpt": "best_b.safetensors",
                    "recipe_ids": ["k0_no_attack", "lora_r8_lr1e5_l5l9"],
                    "attacked_ckpt_by_recipe": {"lora_r8_lr1e5_l5l9": "b_r8.safetensors"},
                    "readout_disruption_flag": "readout_disruption",
                },
            ]
        )
    )
    second.write_text(
        json.dumps(
            [
                {
                    "variant_id": "option_a_base",
                    "initializer_label": "none",
                    "recipe_ids": ["k0_no_attack", "lora_r16_lr5e5_l5l9"],
                    "attacked_ckpt_by_recipe": {"lora_r16_lr5e5_l5l9": "a_r16.safetensors"},
                },
                {
                    "variant_id": "option_b_classification_ce",
                    "initializer_label": "classification_ce",
                    "k0_ckpt": "best_b.safetensors",
                    "recipe_ids": ["k0_no_attack", "lora_r16_lr5e5_l5l9"],
                    "attacked_ckpt_by_recipe": {"lora_r16_lr5e5_l5l9": "b_r16.safetensors"},
                    "readout_disruption_flag": "readout_disruption",
                },
            ]
        )
    )

    merged = merge_variant_rows([str(first), str(second)])

    option_a = next(row for row in merged if row["variant_id"] == "option_a_base")
    option_b = next(row for row in merged if row["variant_id"] == "option_b_classification_ce")
    assert option_a["attacked_ckpt_by_recipe"] == {
        "lora_r16_lr5e5_l5l9": "a_r16.safetensors",
        "lora_r8_lr1e5_l5l9": "a_r8.safetensors",
    }
    assert option_b["attacked_ckpt_by_recipe"] == {
        "lora_r16_lr5e5_l5l9": "b_r16.safetensors",
        "lora_r8_lr1e5_l5l9": "b_r8.safetensors",
    }
    assert option_b["recipe_ids"] == [
        "k0_no_attack",
        "lora_r8_lr1e5_l5l9",
        "lora_r16_lr5e5_l5l9",
    ]


def test_merge_variant_rows_rejects_conflicting_checkpoint_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = [
        {
            "variant_id": "option_a_base",
            "initializer_label": "none",
            "attacked_ckpt_by_recipe": {"lora_r8_lr1e5_l5l9": "a.safetensors"},
        }
    ]
    first.write_text(json.dumps(payload))
    second.write_text(
        json.dumps(
            [
                {
                    "variant_id": "option_a_base",
                    "initializer_label": "none",
                    "attacked_ckpt_by_recipe": {"lora_r8_lr1e5_l5l9": "b.safetensors"},
                }
            ]
        )
    )

    try:
        merge_variant_rows([str(first), str(second)])
    except ValueError as exc:
        assert "conflicting attacked checkpoint paths" in str(exc)
    else:
        raise AssertionError("expected merge_variant_rows to reject conflicting paths")


def test_merge_stage2_script_writes_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps([{"variant_id": "option_a_base", "initializer_label": "none"}]))
    second.write_text(json.dumps([{"variant_id": "option_b_classification_ce", "initializer_label": "classification_ce"}]))
    out_json = tmp_path / "merged.json"

    subprocess.run(
        [
            sys.executable,
            "phase2/merge_stage2_attacked_variants.py",
            "--variant-spec-json",
            str(first),
            "--variant-spec-json",
            str(second),
            "--out-json",
            str(out_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    metadata = json.loads((tmp_path / "merged_metadata.json").read_text())
    assert metadata["phase"] == "merge_stage2_attacked_variants"
    assert metadata["variant_ids"] == ["option_a_base", "option_b_classification_ce"]
