"""Plan the current Stage 2 initializer ablation from Stage 1 artifacts.

This planner does not invent missing attacked checkpoints. It reads a Stage 1
variant spec, keeps only the requested initializer variants, determines which
recipes are truly runnable for each variant, and writes a reduced variant spec
plus a compact readiness report for the current ablation state.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.next_steps_common import DEFAULT_ATTACK_DISTRIBUTION
from phase2.run_metadata import build_run_metadata, write_metadata


def load_variants(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("--variant-spec-json must contain a non-empty JSON list")
    variants: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("variant spec entries must be JSON objects")
        variants.append(dict(item))
    return variants


def normalize_requested_variants(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def variant_recipe_allowlist(variant: dict[str, Any], all_recipes: list[str]) -> list[str]:
    recipe_ids = variant.get("recipe_ids") or []
    if recipe_ids:
        return [str(recipe_id) for recipe_id in recipe_ids if str(recipe_id)]
    return list(all_recipes)


def resolve_recipe_checkpoint(variant: dict[str, Any], recipe_id: str) -> str:
    if recipe_id == "k0_no_attack":
        return str(variant.get("k0_ckpt") or "")
    recipe_map = variant.get("attacked_ckpt_by_recipe") or {}
    if recipe_id in recipe_map:
        return str(recipe_map[recipe_id] or "")
    return str(variant.get("attacked_ckpt") or "")


def recipe_status(variant: dict[str, Any], recipe_id: str) -> tuple[str, str]:
    variant_id = str(variant.get("variant_id") or "")
    if recipe_id == "k0_no_attack":
        ckpt = str(variant.get("k0_ckpt") or "")
        if ckpt:
            return ("ready", ckpt) if Path(ckpt).exists() else ("missing_checkpoint", ckpt)
        if variant_id == "option_a_base":
            return ("ready", "")
        return ("missing_checkpoint", "")
    ckpt = resolve_recipe_checkpoint(variant, recipe_id)
    if not ckpt:
        return ("missing_checkpoint", "")
    return ("ready", ckpt) if Path(ckpt).exists() else ("missing_checkpoint", ckpt)


def summarize_current_results(path: str, requested_variants: list[str], shared_recipes: list[str]) -> list[dict[str, Any]]:
    summary_path = Path(path)
    if not summary_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            variant_id = str(row.get("variant_id") or "")
            recipe_id = str(row.get("recipe_id") or "")
            if variant_id not in requested_variants or recipe_id not in shared_recipes:
                continue
            rows.append(
                {
                    "variant_id": variant_id,
                    "recipe_id": recipe_id,
                    "auroc": row.get("auroc", ""),
                    "metric_excess_over_kmer": row.get("metric_excess_over_kmer", ""),
                    "checkpoint": row.get("checkpoint", ""),
                    "readout_disruption_flag": row.get("readout_disruption_flag", ""),
                    "result_path": row.get("result_path", ""),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant-spec-json",
        default="data/phase2/stage1_variant_specs/stage1_hostonly_smoke_variants.json",
    )
    parser.add_argument(
        "--compare-variants",
        default="option_a_base,option_b_classification_ce",
    )
    parser.add_argument(
        "--existing-summary-csv",
        default="data/phase2/tar_feasibility_smoke_best_optionb_k0_compare/stage1_smoke_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/phase2/stage2_initializer_ablation",
    )
    args = parser.parse_args()

    all_recipes = [recipe.recipe_id for recipe in DEFAULT_ATTACK_DISTRIBUTION]
    requested_variants = normalize_requested_variants(args.compare_variants)
    variants = load_variants(args.variant_spec_json)
    selected_variants = [variant for variant in variants if str(variant.get("variant_id") or "") in requested_variants]
    if len(selected_variants) != len(requested_variants):
        found = {str(variant.get("variant_id") or "") for variant in selected_variants}
        missing = [variant_id for variant_id in requested_variants if variant_id not in found]
        raise ValueError(f"Requested variants not found in spec: {missing}")

    per_variant_status: list[dict[str, Any]] = []
    runnable_recipe_sets: list[set[str]] = []
    reduced_variants: list[dict[str, Any]] = []
    for variant in selected_variants:
        variant_id = str(variant.get("variant_id") or "")
        allowed_recipes = variant_recipe_allowlist(variant, all_recipes)
        recipe_rows: list[dict[str, Any]] = []
        runnable: set[str] = set()
        for recipe_id in allowed_recipes:
            status, ckpt = recipe_status(variant, recipe_id)
            if status == "ready":
                runnable.add(recipe_id)
            recipe_rows.append(
                {
                    "recipe_id": recipe_id,
                    "status": status,
                    "checkpoint": ckpt,
                }
            )
        runnable_recipe_sets.append(runnable)
        reduced_variant = dict(variant)
        reduced_variant["recipe_ids"] = [recipe_id for recipe_id in allowed_recipes if recipe_id in runnable]
        reduced_variants.append(reduced_variant)
        per_variant_status.append(
            {
                "variant_id": variant_id,
                "initializer_label": str(variant.get("initializer_label") or ""),
                "runnable_recipe_ids": reduced_variant["recipe_ids"],
                "recipes": recipe_rows,
            }
        )

    shared_recipes = sorted(set.intersection(*runnable_recipe_sets)) if runnable_recipe_sets else []
    current_results = summarize_current_results(args.existing_summary_csv, requested_variants, shared_recipes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_out = out_dir / "stage2_initializer_ablation_variants.json"
    variant_out.write_text(json.dumps(reduced_variants, indent=2) + "\n")

    report = {
        "source_variant_spec_json": args.variant_spec_json,
        "requested_variants": requested_variants,
        "shared_runnable_recipe_ids": shared_recipes,
        "status": "ready_for_execution" if shared_recipes else "blocked_missing_shared_attacked_checkpoints",
        "per_variant_status": per_variant_status,
        "existing_summary_csv": args.existing_summary_csv,
        "current_comparison_rows": current_results,
    }
    report_out = out_dir / "stage2_initializer_ablation_report.json"
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    plan = {
        "variant_spec_json": str(variant_out),
        "shared_runnable_recipe_ids": shared_recipes,
        "suggested_tar_smoke_command": []
        if not shared_recipes
        else [
            "python",
            "phase2/tar_feasibility_smoke.py",
            "--benchmark-manifest",
            "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv",
            "--tasks",
            "hvue_human_host_tropism",
            "--variant-spec-json",
            str(variant_out),
            "--recipes",
            ",".join(shared_recipes),
            "--out-dir",
            str(out_dir / "tar_feasibility_smoke"),
            "--execute",
        ],
    }
    plan_out = out_dir / "stage2_initializer_ablation_plan.json"
    plan_out.write_text(json.dumps(plan, indent=2) + "\n")
    write_metadata(
        out_dir / "stage2_initializer_ablation_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[args.variant_spec_json, args.existing_summary_csv],
            extra={
                "phase": "plan_stage2_initializer_ablation",
                "variant_out": str(variant_out),
                "report_out": str(report_out),
                "plan_out": str(plan_out),
                "requested_variants": requested_variants,
                "shared_runnable_recipe_ids": shared_recipes,
                "status": report["status"],
            },
        ),
    )

    print(f"[stage2-ablation] wrote {variant_out}")
    print(f"[stage2-ablation] wrote {report_out}")
    print(f"[stage2-ablation] wrote {plan_out}")


if __name__ == "__main__":
    main()
