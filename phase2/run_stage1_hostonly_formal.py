"""Orchestrate the current host-only Stage 1 formal workflow.

This runner sequences the currently supported pieces:
1. source audit
2. formal-target manifest build
3. optional Option B initializer build
4. smoke variant spec build
5. TAR feasibility smoke

It intentionally stays host-only until a valid CINI disjoint source exists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    python_bin = args.python_bin

    plan.append(
        {
            "name": "source_audit",
            "cmd": [
                python_bin,
                "phase2/audit_stage1_target_sources.py",
                "--out-json",
                args.source_audit_json,
            ],
        }
    )
    plan.append(
        {
            "name": "formal_manifest",
            "cmd": [
                python_bin,
                "phase2/build_stage1_formal_target_manifests.py",
                "--out-dir",
                args.formal_manifest_dir,
                "--cini-raw-dir",
                args.cini_raw_dir,
                "--cini-unified-manifest",
                args.cini_unified_manifest,
            ],
        }
    )
    if args.build_option_b:
        plan.append(
            {
                "name": "option_b_initializer",
                "cmd": [
                    python_bin,
                    "phase2/build_stage1_option_b_initializer.py",
                    "--benchmark-manifest",
                    args.formal_manifest_path,
                    "--target-task",
                    args.target_task,
                    "--split-type",
                    args.split_type,
                    "--retain-csv",
                    args.retain_csv,
                    "--target-train-max-rows",
                    str(args.target_train_max_rows),
                    "--target-val-max-rows",
                    str(args.target_val_max_rows),
                    "--target-test-max-rows",
                    str(args.target_test_max_rows),
                    "--retain-max-rows",
                    str(args.retain_max_rows),
                    "--elicitation-steps",
                    str(args.option_b_elicitation_steps),
                    "--ascent-steps",
                    str(args.option_b_ascent_steps),
                    "--eval-every",
                    str(args.option_b_eval_every),
                    "--train-batch-size",
                    str(args.option_b_train_batch_size),
                    "--eval-batch-size",
                    str(args.option_b_eval_batch_size),
                    "--out-dir",
                    args.option_b_out_dir,
                ],
            }
        )
    plan.append(
        {
            "name": "variant_spec",
            "cmd": [
                sys.executable,
                "phase2/build_stage1_smoke_variants.py",
                "--out-dir",
                args.variant_spec_dir,
                "--option-b-k0-ckpt",
                args.option_b_weights_path,
                "--option-b-best-candidate-json",
                args.option_b_best_candidate_json,
            ],
        }
    )
    if not args.skip_smoke:
        smoke_cmd = [
            python_bin,
            "phase2/tar_feasibility_smoke.py",
            "--project-root",
            ".",
            "--python-bin",
            python_bin,
            "--benchmark-manifest",
            args.formal_manifest_path,
            "--tasks",
            args.target_task,
            "--variant-spec-json",
            args.variant_spec_json,
            "--recipes",
            args.smoke_recipes,
            "--validation-max-rows",
            str(args.smoke_validation_max_rows),
            "--test-max-rows",
            str(args.smoke_test_max_rows),
            "--out-dir",
            args.smoke_out_dir,
        ]
        if args.execute_smoke:
            smoke_cmd.append("--execute")
        plan.append({"name": "tar_smoke", "cmd": smoke_cmd})
    return plan


def write_stage1_run_metadata(args: argparse.Namespace, plan: list[dict[str, Any]], preview_path: Path) -> Path:
    metadata_path = preview_path.with_name("stage1_hostonly_formal_metadata.json")
    data_paths = [
        args.formal_manifest_path,
        args.retain_csv,
        args.cini_unified_manifest,
        args.variant_spec_json,
        args.option_b_best_candidate_json,
    ]
    if args.build_option_b:
        data_paths.append(args.option_b_out_dir)
    write_metadata(
        metadata_path,
        build_run_metadata(
            args=args,
            data_paths=data_paths,
            extra={
                "phase": "stage1_hostonly_formal",
                "preview_json": str(preview_path),
                "plan_step_names": [item["name"] for item in plan],
                "plan_command_count": len(plan),
                "target_task": args.target_task,
                "split_type": args.split_type,
                "skip_smoke": bool(args.skip_smoke),
                "execute_smoke": bool(args.execute_smoke),
                "build_option_b": bool(args.build_option_b),
            },
        ),
    )
    return metadata_path


def run_plan(plan: list[dict[str, Any]], cwd: Path) -> None:
    for item in plan:
        print(f"[stage1-hostonly] step={item['name']}", flush=True)
        result = subprocess.run(item["cmd"], cwd=str(cwd), check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--preview-json", default="data/phase2/stage1_hostonly_formal_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-option-b", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--target-task", default="hvue_human_host_tropism")
    parser.add_argument("--split-type", default="cluster_disjoint")
    parser.add_argument("--source-audit-json", default="data/phase2/stage1_formal_target_manifests/stage1_target_source_audit.json")
    parser.add_argument("--formal-manifest-dir", default="data/phase2/stage1_formal_target_manifests")
    parser.add_argument("--formal-manifest-path", default="data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv")
    parser.add_argument("--cini-raw-dir", default="data/benchmarks/raw/hvue/Pathogenecity/CINI")
    parser.add_argument("--cini-unified-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--target-train-max-rows", type=int, default=256)
    parser.add_argument("--target-val-max-rows", type=int, default=128)
    parser.add_argument("--target-test-max-rows", type=int, default=128)
    parser.add_argument("--retain-max-rows", type=int, default=256)
    parser.add_argument("--option-b-elicitation-steps", type=int, default=20)
    parser.add_argument("--option-b-ascent-steps", type=int, default=20)
    parser.add_argument("--option-b-eval-every", type=int, default=5)
    parser.add_argument("--option-b-train-batch-size", type=int, default=4)
    parser.add_argument("--option-b-eval-batch-size", type=int, default=8)
    parser.add_argument("--option-b-out-dir", default="data/phase2/stage1_option_b_initializer/hostonly")
    parser.add_argument("--option-b-weights-path", default="data/phase2/stage1_option_b_initializer/hostonly/weights.safetensors")
    parser.add_argument("--option-b-best-candidate-json", default="data/phase2/stage1_option_b_initializer/best_candidate.json")
    parser.add_argument("--variant-spec-dir", default="data/phase2/stage1_variant_specs")
    parser.add_argument("--variant-spec-json", default="data/phase2/stage1_variant_specs/stage1_hostonly_smoke_variants.json")
    parser.add_argument("--smoke-recipes", default="k0_no_attack,lora_r8_lr1e5_l5l9,full_lr1e5_all")
    parser.add_argument("--smoke-validation-max-rows", type=int, default=128)
    parser.add_argument("--smoke-test-max-rows", type=int, default=256)
    parser.add_argument("--smoke-out-dir", default="data/phase2/tar_feasibility_smoke_formal_targets_hostonly_variants")
    args = parser.parse_args()

    plan = build_plan(args)
    preview_path = PROJECT_ROOT / args.preview_json
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(json.dumps(plan, indent=2) + "\n")
    metadata_path = write_stage1_run_metadata(args, plan, preview_path)
    print(f"[stage1-hostonly] wrote {preview_path}")
    print(f"[stage1-hostonly] wrote {metadata_path}")

    if not args.execute:
        return
    run_plan(plan, PROJECT_ROOT)


if __name__ == "__main__":
    main()
