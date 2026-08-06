"""Minimal TAR feasibility orchestration for Stage 1 smoke runs.

This runner does not implement the full TAR meta-loop yet. It wires together:
1. an attack distribution that explicitly includes K=0;
2. per-recipe downstream fresh-head evaluation via eval_benchmarks.py;
3. result backfilling for post_attack_fresh_head_score.
"""
from __future__ import annotations

import argparse
import csv
import os
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.next_steps_common import DEFAULT_ATTACK_DISTRIBUTION, FORMAL_TARGET_TASKS
from phase2.run_metadata import build_run_metadata, write_metadata


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    attacked_ckpt: str = ""
    attacked_ckpt_by_recipe: dict[str, str] | None = None
    k0_ckpt: str = ""
    initializer_label: str = ""
    readout_disruption_flag: str = ""
    recipe_ids: tuple[str, ...] | None = None


def normalize_variant_id(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "variant"


def parse_tasks(spec: str) -> list[str]:
    if not spec:
        return list(FORMAL_TARGET_TASKS)
    return [part.strip() for part in spec.split(",") if part.strip()]


def parse_recipes(spec: str) -> list[str] | None:
    if not spec:
        return None
    recipes = [part.strip() for part in spec.split(",") if part.strip()]
    return recipes or None


def metric_value(row: dict[str, str]) -> float | None:
    for key in ("auroc", "mcc", "f1", "accuracy", "macro_auroc"):
        value = row.get(key, "")
        if value in ("", "NA", "null"):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def load_kmer_baseline_map(path: str) -> dict[tuple[str, str], float]:
    baseline_map: dict[tuple[str, str], float] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            task = (row.get("task") or "").strip()
            split_type = (row.get("split_type") or "random").strip() or "random"
            if not task:
                continue
            value = metric_value(row)
            if value is None:
                continue
            baseline_map[(task, split_type)] = value
    return baseline_map


def validate_requested_split_type(path: str, tasks: list[str], requested_split_type: str) -> None:
    normalized_requested = requested_split_type.strip().lower()
    if not normalized_requested:
        return
    if normalized_requested in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        normalized_requested = "cluster_disjoint"

    seen_tasks: set[str] = set()
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "split_type" not in fields:
            raise ValueError(
                f"Benchmark manifest {path} does not contain a split_type column, "
                f"so --split-type={normalized_requested} cannot be validated"
            )
        for row in reader:
            task = (row.get("task") or "").strip()
            if task not in tasks:
                continue
            split_type = (row.get("split_type") or "").strip().lower()
            if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
                split_type = "cluster_disjoint"
            if split_type == normalized_requested:
                seen_tasks.add(task)
    missing = [task for task in tasks if task not in seen_tasks]
    if missing:
        raise ValueError(
            f"Benchmark manifest {path} does not contain split_type={normalized_requested} rows for tasks: {missing}"
        )


def load_variant_specs(args: argparse.Namespace) -> list[VariantSpec]:
    if args.variant_spec_json:
        payload = json.loads(Path(args.variant_spec_json).read_text())
        if not isinstance(payload, list) or not payload:
            raise ValueError("--variant-spec-json must contain a non-empty JSON list")
        specs = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"variant entry #{index} must be a JSON object")
            variant_id = normalize_variant_id(str(item.get("variant_id") or item.get("name") or f"variant_{index}"))
            recipe_map = item.get("attacked_ckpt_by_recipe") or {}
            if recipe_map and not isinstance(recipe_map, dict):
                raise ValueError(f"variant {variant_id} attacked_ckpt_by_recipe must be a JSON object")
            specs.append(
                VariantSpec(
                    variant_id=variant_id,
                    attacked_ckpt=str(item.get("attacked_ckpt") or ""),
                    attacked_ckpt_by_recipe={str(k): str(v) for k, v in recipe_map.items()} or None,
                    k0_ckpt=str(item.get("k0_ckpt") or ""),
                    initializer_label=str(item.get("initializer_label") or ""),
                    readout_disruption_flag=str(item.get("readout_disruption_flag") or ""),
                    recipe_ids=tuple(str(value) for value in item.get("recipe_ids", []) if str(value)) or None,
                )
            )
        return specs

    default_id = normalize_variant_id(args.variant_id or "option_a_base")
    return [VariantSpec(variant_id=default_id, attacked_ckpt=args.attacked_ckpt)]


def resolve_checkpoint(recipe_id: str, variant: VariantSpec) -> str:
    if recipe_id == "k0_no_attack":
        return variant.k0_ckpt
    if variant.attacked_ckpt_by_recipe and recipe_id in variant.attacked_ckpt_by_recipe:
        return variant.attacked_ckpt_by_recipe[recipe_id]
    return variant.attacked_ckpt


def build_command(
    python_bin: str,
    project_root: Path,
    args: argparse.Namespace,
    variant: VariantSpec,
    recipe_id: str,
    task_filter: str,
    out_dir: Path,
) -> list[str]:
    ckpt_path = resolve_checkpoint(recipe_id, variant)
    cmd = [
        python_bin,
        "-u",
        str(project_root / "phase2" / "eval_benchmarks.py"),
        "--benchmark-manifest",
        args.benchmark_manifest,
        "--benchmark-scope",
        "task",
        "--task-filter",
        task_filter,
        "--out-dir",
        str(out_dir),
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
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--seed",
        str(args.seed),
        "--split-type",
        args.split_type,
        "--attack-recipe-id",
        recipe_id,
        "--discard-task-checkpoint",
    ]
    if variant.readout_disruption_flag:
        cmd.extend(["--readout-disruption-flag", variant.readout_disruption_flag])
    if ckpt_path:
        cmd[3:3] = ["--ckpt", ckpt_path]
    return cmd


def backfill_fresh_head_score(path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0]) if rows else []
    if not rows or "post_attack_fresh_head_score" not in fieldnames:
        return
    for row in rows:
        if row.get("post_attack_fresh_head_score"):
            continue
        value = metric_value(row)
        if value is not None:
            row["post_attack_fresh_head_score"] = str(value)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backfill_kmer_baseline_scores(path: Path, baseline_map: dict[tuple[str, str], float]) -> None:
    if not path.exists() or not baseline_map:
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0]) if rows else []
    if not rows or "kmer_baseline_score" not in fieldnames or "metric_excess_over_kmer" not in fieldnames:
        return

    changed = False
    for row in rows:
        task = (row.get("task") or "").strip()
        split_type = (row.get("split_type") or "random").strip() or "random"
        baseline = baseline_map.get((task, split_type))
        if baseline is None:
            continue
        score = metric_value(row)
        row["kmer_baseline_score"] = str(baseline)
        row["metric_excess_over_kmer"] = "" if score is None else str(score - baseline)
        changed = True
    if not changed:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(commands: list[dict[str, Any]], out_dir: Path) -> Path | None:
    rows: list[dict[str, Any]] = []
    for item in commands:
        result_path = Path(item["out_dir"]) / "eval_benchmarks.csv"
        if not result_path.exists():
            continue
        with result_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "variant_id": item["variant_id"],
                        "initializer_label": item["initializer_label"],
                        "recipe_id": item["recipe_id"],
                        "task": row.get("task", ""),
                        "checkpoint": row.get("checkpoint", ""),
                        "split_type": row.get("split_type", ""),
                        "kmer_baseline_score": row.get("kmer_baseline_score", ""),
                        "metric_excess_over_kmer": row.get("metric_excess_over_kmer", ""),
                        "auroc": row.get("auroc", ""),
                        "mcc": row.get("mcc", ""),
                        "accuracy": row.get("accuracy", ""),
                        "post_attack_fresh_head_score": row.get("post_attack_fresh_head_score", ""),
                        "readout_disruption_flag": row.get("readout_disruption_flag", ""),
                        "n_test_eval": row.get("n_test_eval", ""),
                        "result_path": str(result_path),
                    }
                )
    if not rows:
        return None

    summary_path = out_dir / "stage1_smoke_summary.csv"
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def apply_backfills(commands: list[dict[str, Any]], baseline_map: dict[tuple[str, str], float]) -> None:
    for item in commands:
        result_path = Path(item["out_dir"]) / "eval_benchmarks.csv"
        backfill_fresh_head_score(result_path)
        backfill_kmer_baseline_scores(result_path, baseline_map)


def write_smoke_run_metadata(
    args: argparse.Namespace,
    out_dir: Path,
    tasks: list[str],
    variants: list[VariantSpec],
    commands: list[dict[str, Any]],
) -> Path:
    metadata_path = out_dir / "run_metadata.json"
    data_paths = [args.benchmark_manifest]
    if args.kmer_baseline_csv:
        data_paths.append(args.kmer_baseline_csv)
    if args.variant_spec_json:
        data_paths.append(args.variant_spec_json)
    write_metadata(
        metadata_path,
        build_run_metadata(
            args=args,
            data_paths=data_paths,
            extra={
                "phase": "tar_feasibility_smoke",
                "target_tasks": tasks,
                "split_type": args.split_type,
                "variant_ids": [variant.variant_id for variant in variants],
                "initializer_labels": [variant.initializer_label for variant in variants],
                "recipe_ids": [item["recipe_id"] for item in commands],
                "command_count": len(commands),
                "execute": bool(args.execute),
                "backfill_only": bool(args.backfill_only),
            },
        ),
    )
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--attacked-ckpt", default="")
    parser.add_argument("--variant-spec-json", default="")
    parser.add_argument("--variant-id", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--recipes", default="")
    parser.add_argument("--kmer-baseline-csv", default="")
    parser.add_argument("--out-dir", default="data/phase2/tar_feasibility_smoke")
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
    parser.add_argument("--test-max-rows", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backfill-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    task_filter = ",".join(tasks)
    variants = load_variant_specs(args)
    recipe_filter = parse_recipes(args.recipes)
    baseline_map = load_kmer_baseline_map(args.kmer_baseline_csv) if args.kmer_baseline_csv else {}
    validate_requested_split_type(args.benchmark_manifest, tasks, args.split_type)

    recipes = [recipe.recipe_id for recipe in DEFAULT_ATTACK_DISTRIBUTION]
    if recipe_filter is not None:
        unknown = sorted(set(recipe_filter) - set(recipes))
        if unknown:
            raise ValueError(f"Unknown recipe ids: {unknown}")
        recipes = [recipe_id for recipe_id in recipes if recipe_id in set(recipe_filter)]
    commands = []
    for variant in variants:
        for recipe_id in recipes:
            if variant.recipe_ids is not None and recipe_id not in variant.recipe_ids:
                continue
            recipe_out = out_dir / variant.variant_id / recipe_id
            commands.append(
                {
                    "variant_id": variant.variant_id,
                    "initializer_label": variant.initializer_label,
                    "recipe_id": recipe_id,
                    "ckpt_path": resolve_checkpoint(recipe_id, variant),
                    "out_dir": str(recipe_out),
                    "command": build_command(args.python_bin, project_root, args, variant, recipe_id, task_filter, recipe_out),
                }
            )

    commands_path = out_dir / "commands.json"
    commands_path.write_text(
        json.dumps(
            {
                "tasks": tasks,
                "variants": [
                    {
                        "variant_id": variant.variant_id,
                        "initializer_label": variant.initializer_label,
                        "attacked_ckpt": variant.attacked_ckpt,
                        "attacked_ckpt_by_recipe": variant.attacked_ckpt_by_recipe or {},
                        "k0_ckpt": variant.k0_ckpt,
                        "readout_disruption_flag": variant.readout_disruption_flag,
                        "recipe_ids": list(variant.recipe_ids or []),
                    }
                    for variant in variants
                ],
                "recipes": commands,
            },
            indent=2,
        )
        + "\n"
    )
    metadata_path = write_smoke_run_metadata(args, out_dir, tasks, variants, commands)
    print(f"[tar-smoke] wrote {commands_path}")
    print(f"[tar-smoke] wrote {metadata_path}")

    if not args.execute and not args.backfill_only:
        return

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(project_root)
    if args.execute:
        for item in commands:
            print(f"[tar-smoke] run variant={item['variant_id']} recipe={item['recipe_id']}", flush=True)
            result = subprocess.run(item["command"], cwd=str(project_root), check=False, env=env)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
    apply_backfills(commands, baseline_map)
    summary_path = summarize_results(commands, out_dir)
    if summary_path is not None:
        print(f"[tar-smoke] wrote {summary_path}")
    if args.execute:
        print(f"[tar-smoke] completed {len(commands)} recipe runs")
    elif args.backfill_only:
        print(f"[tar-smoke] completed backfill for {len(commands)} recipe entries")


if __name__ == "__main__":
    main()
