from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.run_stage1_hostonly_formal import build_plan, write_stage1_run_metadata


def make_args(**overrides):
    defaults = dict(
        python_bin="python",
        preview_json="plan.json",
        execute=False,
        build_option_b=False,
        skip_smoke=False,
        execute_smoke=False,
        target_task="hvue_human_host_tropism",
        split_type="cluster_disjoint",
        source_audit_json="audit.json",
        formal_manifest_dir="formal_dir",
        formal_manifest_path="formal.csv",
        cini_raw_dir="cini_raw",
        cini_unified_manifest="cini_manifest.csv",
        retain_csv="retain.csv",
        target_train_max_rows=256,
        target_val_max_rows=128,
        target_test_max_rows=128,
        retain_max_rows=256,
        option_b_elicitation_steps=20,
        option_b_ascent_steps=20,
        option_b_eval_every=5,
        option_b_train_batch_size=4,
        option_b_eval_batch_size=8,
        option_b_out_dir="option_b_out",
        option_b_weights_path="option_b_out/weights.safetensors",
        option_b_best_candidate_json="option_b_out/best_candidate.json",
        variant_spec_dir="variants",
        variant_spec_json="variants/spec.json",
        smoke_recipes="k0_no_attack,lora_r8_lr1e5_l5l9",
        smoke_validation_max_rows=128,
        smoke_test_max_rows=256,
        smoke_out_dir="smoke_out",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_plan_includes_core_steps() -> None:
    plan = build_plan(make_args())
    assert [step["name"] for step in plan] == [
        "source_audit",
        "formal_manifest",
        "variant_spec",
        "tar_smoke",
    ]


def test_build_plan_can_include_option_b_step() -> None:
    plan = build_plan(make_args(build_option_b=True))
    assert [step["name"] for step in plan] == [
        "source_audit",
        "formal_manifest",
        "option_b_initializer",
        "variant_spec",
        "tar_smoke",
    ]
    option_b_cmd = plan[2]["cmd"]
    assert "--benchmark-manifest" in option_b_cmd
    assert "formal.csv" in option_b_cmd
    variant_cmd = plan[3]["cmd"]
    assert "--option-b-best-candidate-json" in variant_cmd
    assert "option_b_out/best_candidate.json" in variant_cmd


def test_build_plan_can_skip_smoke() -> None:
    plan = build_plan(make_args(skip_smoke=True))
    assert [step["name"] for step in plan] == [
        "source_audit",
        "formal_manifest",
        "variant_spec",
    ]


def test_build_plan_execute_smoke_appends_flag() -> None:
    plan = build_plan(make_args(execute_smoke=True))
    assert plan[-1]["cmd"][-1] == "--execute"


def test_write_stage1_run_metadata_records_plan(tmp_path: Path) -> None:
    args = make_args(
        preview_json=str(tmp_path / "plan.json"),
        formal_manifest_path=str(tmp_path / "formal.csv"),
        retain_csv=str(tmp_path / "retain.csv"),
        cini_unified_manifest=str(tmp_path / "cini.csv"),
        variant_spec_json=str(tmp_path / "variants.json"),
        option_b_best_candidate_json=str(tmp_path / "best.json"),
    )
    for path in [
        Path(args.formal_manifest_path),
        Path(args.retain_csv),
        Path(args.cini_unified_manifest),
        Path(args.variant_spec_json),
        Path(args.option_b_best_candidate_json),
    ]:
        path.write_text("placeholder\n")
    plan = build_plan(args)
    preview_path = Path(args.preview_json)
    preview_path.write_text("[]\n")

    metadata_path = write_stage1_run_metadata(args, plan, preview_path)

    payload = json.loads(metadata_path.read_text())
    assert payload["phase"] == "stage1_hostonly_formal"
    assert payload["target_task"] == "hvue_human_host_tropism"
    assert payload["plan_step_names"] == ["source_audit", "formal_manifest", "variant_spec", "tar_smoke"]
