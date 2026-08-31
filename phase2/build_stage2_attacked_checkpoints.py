"""Build attacked checkpoints for current Stage 2 initializer ablations.

This runner materializes post-attack checkpoint artifacts from the same
supervised target-task training loop used by eval_benchmarks.py. It currently
supports LoRA attack recipes and emits explicit blocked status for full-FT
recipes until a full-finetune export path is wired in.
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

from phase2.next_steps_common import DEFAULT_ATTACK_DISTRIBUTION
from phase2.run_metadata import build_run_metadata, write_metadata
from phase2.project_python import project_python


DEFAULT_PROJECT_PYTHON = project_python()


def load_variants(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("--variant-spec-json must contain a non-empty JSON list")
    return [dict(item) for item in payload]


def parse_requested_recipes(spec: str) -> list[str]:
    if not spec.strip():
        return [recipe.recipe_id for recipe in DEFAULT_ATTACK_DISTRIBUTION if recipe.recipe_id != "k0_no_attack"]
    return [part.strip() for part in spec.split(",") if part.strip()]


def format_layers(layers: tuple[int, ...]) -> str:
    if not layers:
        return ""
    ordered = sorted(set(int(layer) for layer in layers))
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(layer) for layer in ordered)


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    recipe_map = {recipe.recipe_id: recipe for recipe in DEFAULT_ATTACK_DISTRIBUTION}
    variants = load_variants(args.variant_spec_json)
    commands: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant.get("variant_id") or "")
        initializer_label = str(variant.get("initializer_label") or "")
        init_ckpt = str(variant.get("k0_ckpt") or "")
        variant_recipe_ids = variant.get("recipe_ids") or []
        if getattr(args, "respect_variant_recipe_ids", False) and variant_recipe_ids:
            allowed = {str(recipe_id) for recipe_id in variant_recipe_ids}
        else:
            allowed = {recipe.recipe_id for recipe in DEFAULT_ATTACK_DISTRIBUTION}
        for recipe_id in parse_requested_recipes(args.recipes):
            if recipe_id not in recipe_map:
                raise ValueError(f"Unknown attack recipe id: {recipe_id}")
            if recipe_id not in allowed:
                commands.append(
                    {
                        "variant_id": variant_id,
                        "initializer_label": initializer_label,
                        "recipe_id": recipe_id,
                        "status": "skipped_by_variant_recipe_ids",
                        "command": [],
                    }
                )
                continue
            recipe = recipe_map[recipe_id]
            if recipe.full_ft or recipe.method != "lora_ft":
                commands.append(
                    {
                        "variant_id": variant_id,
                        "initializer_label": initializer_label,
                        "recipe_id": recipe_id,
                        "status": "blocked_unsupported_attack_method",
                        "reason": "Current attacked-checkpoint materialization runner supports LoRA attack recipes only",
                        "command": [],
                    }
                )
                continue
            recipe_out = Path(args.out_dir) / variant_id / recipe_id
            export_dir = recipe_out / "exported_attack"
            cmd = [
                args.python_bin,
                "phase2/eval_benchmarks.py",
                "--benchmark-manifest",
                args.benchmark_manifest,
                "--benchmark-scope",
                "task",
                "--task-filter",
                args.target_task,
                "--out-dir",
                str(recipe_out / "eval"),
                "--resume",
                "--device",
                args.device,
                "--cpu-threads",
                str(args.cpu_threads),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--max-length",
                str(args.max_length),
                "--epochs",
                str(args.epochs),
                "--max-steps",
                str(args.max_steps),
                "--eval-every",
                str(args.eval_every),
                "--validation-max-rows",
                str(args.validation_max_rows),
                "--test-max-rows",
                str(args.test_max_rows),
                "--lr",
                str(recipe.lr),
                "--lora-rank",
                str(recipe.rank),
                "--lora-alpha",
                str(args.lora_alpha),
                "--lora-dropout",
                str(args.lora_dropout),
                "--seed",
                str(args.seed),
                "--split-type",
                args.split_type,
                "--attack-recipe-id",
                recipe.recipe_id,
                "--discard-task-checkpoint",
                "--export-attack-ckpt-dir",
                str(export_dir),
                "--export-attack-policy",
                args.export_attack_policy,
                "--export-attack-layers",
                format_layers(recipe.target_layers),
                "--export-attack-suffixes",
                args.export_attack_suffixes,
            ]
            if args.metric_for_best:
                cmd.extend(["--metric-for-best", args.metric_for_best])
            if init_ckpt:
                cmd.extend(["--ckpt", init_ckpt])
            readout_flag = str(variant.get("readout_disruption_flag") or "")
            if readout_flag:
                cmd.extend(["--readout-disruption-flag", readout_flag])
            commands.append(
                {
                    "variant_id": variant_id,
                    "initializer_label": initializer_label,
                    "recipe_id": recipe_id,
                    "status": "ready",
                    "init_ckpt": init_ckpt,
                    "expected_exported_weights": str(export_dir / args.target_task / "weights.safetensors"),
                    "command": cmd,
                }
            )
    return commands


def update_variant_spec(
    variants_path: str,
    commands: list[dict[str, Any]],
    out_path: Path,
) -> None:
    variants = load_variants(variants_path)
    ready_map: dict[tuple[str, str], str] = {}
    for item in commands:
        if item.get("status") != "ready":
            continue
        ready_map[(str(item["variant_id"]), str(item["recipe_id"]))] = str(item["expected_exported_weights"])
    for variant in variants:
        recipe_map = dict(variant.get("attacked_ckpt_by_recipe") or {})
        recipe_ids = [str(recipe_id) for recipe_id in (variant.get("recipe_ids") or []) if str(recipe_id)]
        for (variant_id, recipe_id), weights_path in ready_map.items():
            if str(variant.get("variant_id") or "") != variant_id:
                continue
            recipe_map[recipe_id] = weights_path
            if recipe_id not in recipe_ids:
                recipe_ids.append(recipe_id)
        variant["attacked_ckpt_by_recipe"] = recipe_map
        if recipe_ids:
            variant["recipe_ids"] = recipe_ids
    out_path.write_text(json.dumps(variants, indent=2) + "\n")


def summarize(commands: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for item in commands:
        row = {
            "variant_id": item.get("variant_id", ""),
            "initializer_label": item.get("initializer_label", ""),
            "recipe_id": item.get("recipe_id", ""),
            "status": item.get("status", ""),
            "expected_exported_weights": item.get("expected_exported_weights", ""),
            "init_ckpt": item.get("init_ckpt", ""),
            "reason": item.get("reason", ""),
        }
        rows.append(row)
    (out_dir / "stage2_attacked_checkpoint_summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def write_stage2_run_metadata(args: argparse.Namespace, commands: list[dict[str, Any]], out_dir: Path) -> Path:
    metadata_path = out_dir / "stage2_attacked_checkpoint_metadata.json"
    write_metadata(
        metadata_path,
        build_run_metadata(
            args=args,
            data_paths=[args.variant_spec_json, args.benchmark_manifest],
            extra={
                "phase": "stage2_attacked_checkpoint_build",
                "target_task": args.target_task,
                "split_type": args.split_type,
                "recipe_ids_requested": parse_requested_recipes(args.recipes),
                "command_count": len(commands),
                "ready_count": sum(1 for item in commands if item.get("status") == "ready"),
                "blocked_count": sum(1 for item in commands if item.get("status") != "ready"),
                "variant_ids": sorted({str(item.get("variant_id") or "") for item in commands}),
            },
        ),
    )
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=DEFAULT_PROJECT_PYTHON)
    parser.add_argument(
        "--variant-spec-json",
        default="data/phase2/stage2_initializer_ablation/stage2_initializer_ablation_variants.json",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv",
    )
    parser.add_argument("--target-task", default="hvue_human_host_tropism")
    parser.add_argument("--recipes", default="lora_r8_lr1e5_l5l9")
    parser.add_argument("--respect-variant-recipe-ids", action="store_true")
    parser.add_argument("--split-type", default="cluster_disjoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--validation-max-rows", type=int, default=128)
    parser.add_argument("--test-max-rows", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--metric-for-best", default="auto")
    parser.add_argument("--export-attack-policy", choices=["selected_modules", "delta", "full"], default="delta")
    parser.add_argument("--export-attack-suffixes", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/phase2/stage2_attacked_checkpoints")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    (out_dir / "stage2_attacked_checkpoint_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    update_variant_spec(
        args.variant_spec_json,
        commands,
        out_dir / "stage2_variants_with_attacked_ckpts.json",
    )
    summarize(commands, out_dir)
    metadata_path = write_stage2_run_metadata(args, commands, out_dir)
    print(f"[stage2-attack-build] wrote {out_dir / 'stage2_attacked_checkpoint_commands.json'}")
    print(f"[stage2-attack-build] wrote {out_dir / 'stage2_variants_with_attacked_ckpts.json'}")
    print(f"[stage2-attack-build] wrote {out_dir / 'stage2_attacked_checkpoint_summary.json'}")
    print(f"[stage2-attack-build] wrote {metadata_path}")

    if not args.execute:
        return
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(PROJECT_ROOT)
    for item in commands:
        if item.get("status") != "ready":
            continue
        print(f"[stage2-attack-build] run variant={item['variant_id']} recipe={item['recipe_id']}", flush=True)
        result = subprocess.run(item["command"], cwd=str(PROJECT_ROOT), check=False, env=env)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
