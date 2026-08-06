"""Build reusable Stage 1 smoke variant specs from available checkpoints.

This does not invent missing TAR initializers. It packages the currently
available host-tropism checkpoint controls into a variant-spec JSON file and
records which planned initializer families are still unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata

def variant_entry(
    *,
    variant_id: str,
    initializer_label: str,
    k0_ckpt: str = "",
    attacked_ckpt: str = "",
    attacked_ckpt_by_recipe: dict[str, str] | None = None,
    readout_disruption_flag: str = "",
) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "initializer_label": initializer_label,
        "k0_ckpt": k0_ckpt,
        "attacked_ckpt": attacked_ckpt,
        "attacked_ckpt_by_recipe": attacked_ckpt_by_recipe or {},
        "readout_disruption_flag": readout_disruption_flag,
    }


def must_exist(path: str) -> str:
    if path and not Path(path).exists():
        raise FileNotFoundError(path)
    return path


def resolve_option_b_k0_ckpt(explicit_path: str, best_candidate_json: str) -> tuple[str, dict[str, Any] | None]:
    explicit = Path(explicit_path)
    if explicit.exists():
        return str(explicit), None
    best_path = Path(best_candidate_json)
    if not best_path.exists():
        return explicit_path, None
    report = json.loads(best_path.read_text())
    best = report.get("best_candidate") or {}
    weights_path = str(best.get("weights_path", "") or "")
    if not weights_path:
        return explicit_path, report
    resolved = must_exist(weights_path)
    return resolved, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/phase2/stage1_variant_specs")
    parser.add_argument("--base-variant-id", default="option_a_base")
    parser.add_argument("--projinit-variant-id", default="legacy_projinit_control")
    parser.add_argument("--projinit-k0-ckpt", default="data/phase2/checkpoints_tuned/refseq_gd_projinit_random_ar5_s1000/weights.safetensors")
    parser.add_argument("--projinit-lora-ckpt", default="data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s500/weights.safetensors")
    parser.add_argument("--projinit-fallback-ckpt", default="data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors")
    parser.add_argument("--projinit-full-ckpt", default="data/phase2/checkpoints_tuned/refseq_gd_projinit_full_ar5_s200/weights.safetensors")
    parser.add_argument("--option-b-k0-ckpt", default="data/phase2/stage1_option_b_initializer/hostonly/weights.safetensors")
    parser.add_argument("--option-b-best-candidate-json", default="data/phase2/stage1_option_b_initializer/best_candidate.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_variant = variant_entry(
        variant_id=args.base_variant_id,
        initializer_label="none",
    )

    projinit_variant = variant_entry(
        variant_id=args.projinit_variant_id,
        initializer_label="probe_nullspace_projinit_control",
        k0_ckpt=must_exist(args.projinit_k0_ckpt),
        attacked_ckpt=must_exist(args.projinit_fallback_ckpt),
        attacked_ckpt_by_recipe={
            "lora_r8_lr1e5_l5l9": must_exist(args.projinit_lora_ckpt),
            "full_lr1e5_all": must_exist(args.projinit_full_ckpt),
        },
        readout_disruption_flag="legacy_initializer_control",
    )

    variants = [base_variant, projinit_variant]
    resolved_option_b_k0_ckpt, best_candidate_report = resolve_option_b_k0_ckpt(
        args.option_b_k0_ckpt,
        args.option_b_best_candidate_json,
    )
    option_b_available = Path(resolved_option_b_k0_ckpt).exists()
    if option_b_available:
        variants.append(
            {
                "variant_id": "option_b_classification_ce",
                "initializer_label": "classification_ce",
                "k0_ckpt": must_exist(resolved_option_b_k0_ckpt),
                "attacked_ckpt": "",
                "attacked_ckpt_by_recipe": {},
                "readout_disruption_flag": "readout_disruption",
                "recipe_ids": ["k0_no_attack"],
            }
        )
    variants_path = out_dir / "stage1_hostonly_smoke_variants.json"
    variants_path.write_text(json.dumps(variants, indent=2) + "\n")

    report = {
        "variant_spec_json": str(variants_path),
        "available_variants": [
            {
                "variant_id": base_variant["variant_id"],
                "status": "available",
                "notes": "Base TAR smoke control with no initializer checkpoint",
            },
            {
                "variant_id": projinit_variant["variant_id"],
                "status": "available",
                "notes": "Legacy projection-initialized GD control packaged for Stage 1 smoke comparisons",
                "k0_ckpt": projinit_variant["k0_ckpt"],
                "attacked_ckpt": projinit_variant["attacked_ckpt"],
                "attacked_ckpt_by_recipe": projinit_variant["attacked_ckpt_by_recipe"],
            },
        ],
        "missing_planned_variants": [
            {
                "variant_id": "option_c_lora_subspace_depriming",
                "status": "missing",
                "reason": "No checked-in LoRA-subspace targeting initializer artifact is available yet",
            },
            {
                "variant_id": "option_d_anti_finetune_displacement",
                "status": "missing",
                "reason": "No checked-in anti-fine-tune displacement initializer artifact is available yet",
            },
        ],
    }
    if option_b_available:
        option_b_notes = "Generated Option B classification-CE initializer checkpoint; currently wired for k0_no_attack only"
        if best_candidate_report is not None:
            option_b_notes = (
                "Selected Option B best candidate auto-resolved from best_candidate.json; "
                "currently wired for k0_no_attack only"
            )
        report["available_variants"].append(
            {
                "variant_id": "option_b_classification_ce",
                "status": "available",
                "notes": option_b_notes,
                "k0_ckpt": resolved_option_b_k0_ckpt,
                "recipe_ids": ["k0_no_attack"],
                "best_candidate_json": args.option_b_best_candidate_json if best_candidate_report is not None else "",
                "best_candidate_config_id": ""
                if best_candidate_report is None
                else (best_candidate_report.get("best_candidate") or {}).get("config_id", ""),
            }
        )
    else:
        report["missing_planned_variants"].insert(
            0,
            {
                "variant_id": "option_b_classification_ce",
                "status": "missing",
                "reason": "No checked-in classification-CE initializer checkpoint bundle is available yet",
            },
        )
    report_path = out_dir / "stage1_hostonly_smoke_variants_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_metadata(
        out_dir / "stage1_hostonly_smoke_variants_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[
                args.projinit_k0_ckpt,
                args.projinit_lora_ckpt,
                args.projinit_fallback_ckpt,
                args.projinit_full_ckpt,
                args.option_b_k0_ckpt,
                args.option_b_best_candidate_json,
            ],
            extra={
                "phase": "build_stage1_smoke_variants",
                "variant_spec_json": str(variants_path),
                "report_path": str(report_path),
                "available_variant_ids": [row["variant_id"] for row in variants],
                "missing_planned_variant_ids": [row["variant_id"] for row in report["missing_planned_variants"]],
            },
        ),
    )

    print(f"[stage1-variants] wrote {variants_path}")
    print(f"[stage1-variants] wrote {report_path}")


if __name__ == "__main__":
    main()
