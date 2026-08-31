"""Launch the current Stage 1 formal probe-vs-SFT / LoRA experiment tranche.

This runner is intentionally conservative:
1. it reads the formal-target manifest report rather than inventing task scope;
2. it only launches tasks with an explicit available formal manifest;
3. it records blocked formal targets separately so host-only execution is explicit.

The current repository state exposes a validated cluster-disjoint host-tropism
formal target, while CINI remains blocked. The launcher therefore starts the
host-tropism tranche now and leaves blocked tasks visible in metadata.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import csv
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata
from phase2.project_python import project_python


DEFAULT_PROJECT_PYTHON = project_python()
FORMAL_CHECKPOINTS = {
    "projection_rank32": "data/phase2/checkpoints_projection_adaptive_rank32/projopt_host5_9_coro0_10_adaptive_basis_rank32/weights.safetensors",
    "gd_loc_s1000": "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors",
    "rmu_joint_sc50_ar5": "data/phase2/checkpoints_rmu_localized_joint_probe/rmu_loc_l5_l9_jointprobe_sc50_ar5_s500/weights.safetensors",
    "gd_full_control": "data/phase2/checkpoints_tuned/refseq_gd_projinit_full_ar5_s200/weights.safetensors",
}
DEFAULT_OPTION_B_BEST = "data/phase2/stage1_option_b_initializer/best_candidate.json"
DEFAULT_FORMAL_REPORT = "data/phase2/stage1_formal_target_manifests/stage1_formal_target_manifest_report.json"
DEFAULT_KMER_BASELINE = "data/phase2/kmer_baselines/stage1_formal_targets_available_kmer.csv"
DEFAULT_OUT_ROOT = "data/phase2/stage1_formal_experiment_20260727"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def lr_tag(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")


def resolve_option_b_checkpoint(best_candidate_json: Path) -> tuple[str, str]:
    report = load_json(best_candidate_json)
    best = report.get("best_candidate") or {}
    weights_path = str(best.get("weights_path", "")).strip()
    if not weights_path:
        raise ValueError(f"No best_candidate.weights_path found in {best_candidate_json}")
    config_id = str(best.get("config_id", "")).strip() or "option_b_best"
    return config_id, weights_path


def normalize_split_type(value: str) -> str:
    split_type = str(value or "").strip().lower()
    if not split_type:
        return "random"
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type


def available_formal_tasks(report_path: Path) -> tuple[list[str], list[dict[str, Any]], str]:
    report = load_json(report_path)
    available = sorted(report.get("manifests", {}).keys())
    blocked = list(report.get("missing_targets", []))
    merged_manifest = str(report.get("merged_manifest", "")).strip()
    if not merged_manifest:
        raise ValueError(f"Formal report {report_path} does not define merged_manifest")
    return available, blocked, merged_manifest


def load_kmer_baseline_map(path: Path) -> dict[tuple[str, str], float]:
    baseline_map: dict[tuple[str, str], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = str(row.get("task", "")).strip()
            if not task:
                continue
            split_type = normalize_split_type(str(row.get("split_type", "")))
            value = str(row.get("auroc", "")).strip()
            if not value:
                continue
            baseline_map[(task, split_type)] = float(value)
    return baseline_map


def checkpoint_specs(
    best_candidate_json: Path,
    *,
    include_modified: bool = True,
) -> tuple[list[dict[str, str]], list[str]]:
    specs = [{"name": "base", "path": ""}]
    missing: list[str] = []
    if not include_modified:
        return specs, missing
    for name, path in FORMAL_CHECKPOINTS.items():
        abs_path = PROJECT_ROOT / path
        if abs_path.exists():
            specs.append({"name": name, "path": path})
        else:
            missing.append(name)
    if best_candidate_json.exists():
        option_b_name, option_b_path = resolve_option_b_checkpoint(best_candidate_json)
        option_b_abs = PROJECT_ROOT / option_b_path
        if option_b_abs.exists():
            specs.append({"name": f"option_b_{option_b_name}", "path": option_b_path})
        else:
            missing.append(f"option_b_{option_b_name}")
    else:
        missing.append("option_b_best_candidate_json_missing")
    return specs, missing


def checkpoints_arg(specs: list[dict[str, str]]) -> str:
    parts = []
    for spec in specs:
        if not spec["path"]:
            continue
        parts.append(f"{spec['name']}={spec['path']}")
    return ",".join(parts)


def build_probe_vs_sft_commands(
    args: argparse.Namespace,
    *,
    merged_manifest: str,
    tasks: list[str],
    specs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    checkpoint_spec = checkpoints_arg(specs)
    for lr in args.classification_head_lrs:
        out_dir = Path(args.out_root) / "probe_vs_sft" / f"lr_{lr_tag(lr)}"
        commands.append(
            {
                "name": f"probe_vs_sft_lr_{lr_tag(lr)}",
                "family": "probe_vs_sft",
                "lr": lr,
                "cmd": [
                    args.python_bin,
                    "-u",
                    "phase2/probe_vs_sft.py",
                    "--benchmark-manifest",
                    merged_manifest,
                    "--tasks",
                    ",".join(tasks),
                    "--checkpoints",
                    checkpoint_spec,
                    "--seeds",
                    ",".join(str(seed) for seed in args.seeds),
                    "--out-dir",
                    str(out_dir),
                    "--feature-cache-dir",
                    str(out_dir.parent / "feature_cache"),
                    "--device",
                    args.device,
                    "--layers",
                    args.layers,
                    "--sft-layer",
                    str(args.sft_layer),
                    "--batch-size",
                    str(args.classification_head_batch_size),
                    "--feature-batch-size",
                    str(args.feature_batch_size),
                    "--auto-batch-size",
                    str(args.auto_batch_size),
                    "--max-length",
                    str(args.max_length),
                    "--sft-steps",
                    str(args.sft_steps),
                    "--eval-every",
                    str(args.sft_eval_every),
                    "--patience",
                    str(args.sft_patience),
                    "--lr",
                    str(lr),
                    "--probe-jobs",
                    str(args.probe_jobs),
                    "--cpu-threads",
                    str(args.cpu_threads),
                ],
            }
        )
    return commands


def build_lora_commands(
    args: argparse.Namespace,
    *,
    merged_manifest: str,
    tasks: list[str],
    specs: list[dict[str, str]],
    baseline_map: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    baseline_score = None
    if len(tasks) == 1:
        baseline_score = baseline_map.get((tasks[0], normalize_split_type(args.split_type)))
    for spec in specs:
        for rank in args.lora_ranks:
            alpha = rank * 2
            for lr in args.lora_lrs:
                for seed in args.seeds:
                    out_dir = (
                        Path(args.out_root)
                        / "fresh_lora"
                        / spec["name"]
                        / f"rank_{rank}"
                        / f"lr_{lr_tag(lr)}"
                        / f"seed_{seed}"
                    )
                    cmd = [
                        args.python_bin,
                        "-u",
                        "phase2/eval_benchmarks.py",
                        "--benchmark-manifest",
                        merged_manifest,
                        "--benchmark-scope",
                        "task",
                        "--task-filter",
                        ",".join(tasks),
                        "--out-dir",
                        str(out_dir),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.lora_epochs),
                        "--max-steps",
                        str(args.lora_max_steps),
                        "--eval-every",
                        str(args.lora_eval_every),
                        "--validation-max-rows",
                        str(args.validation_max_rows),
                        "--test-max-rows",
                        str(args.test_max_rows),
                        "--lr",
                        str(lr),
                        "--lora-rank",
                        str(rank),
                        "--lora-alpha",
                        str(alpha),
                        "--lora-dropout",
                        str(args.lora_dropout),
                        "--train-batch-size",
                        str(args.train_batch_size),
                        "--eval-batch-size",
                        str(args.eval_batch_size),
                        "--max-length",
                        str(args.max_length),
                        "--device",
                        args.device,
                        "--cpu-threads",
                        str(args.cpu_threads),
                        "--metric-for-best",
                        args.metric_for_best,
                        "--split-type",
                        args.split_type,
                        "--discard-task-checkpoint",
                        "--resume",
                    ]
                    if baseline_score is not None:
                        cmd.extend(["--kmer-baseline-score", str(baseline_score)])
                    if spec["path"]:
                        cmd[3:3] = ["--ckpt", spec["path"]]
                    commands.append(
                        {
                            "name": f"fresh_lora_{spec['name']}_r{rank}_lr{lr_tag(lr)}_seed{seed}",
                            "family": "fresh_lora",
                            "checkpoint": spec["name"],
                            "rank": rank,
                            "lr": lr,
                            "seed": seed,
                            "cmd": cmd,
                        }
                    )
    return commands


def write_plan(
    out_root: Path,
    *,
    args: argparse.Namespace,
    commands: list[dict[str, Any]],
    tasks: list[str],
    blocked_tasks: list[dict[str, Any]],
    specs: list[dict[str, str]],
    missing_checkpoints: list[str],
    merged_manifest: str,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    plan_path = out_root / "stage1_formal_experiment_plan.json"
    plan_path.write_text(json.dumps(commands, indent=2) + "\n")
    write_metadata(
        out_root / "stage1_formal_experiment_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[
                str(PROJECT_ROOT / args.formal_report_json),
                str(PROJECT_ROOT / merged_manifest),
                str(PROJECT_ROOT / args.option_b_best_candidate_json),
                *[
                    str(PROJECT_ROOT / spec["path"])
                    for spec in specs
                    if spec["path"]
                ],
            ],
            extra={
                "phase": "stage1_formal_experiment",
                "out_root": str(out_root),
                "merged_manifest": merged_manifest,
                "available_tasks": tasks,
                "blocked_tasks": blocked_tasks,
                "checkpoint_mode": args.checkpoint_mode,
                "include_probe_vs_sft": bool(args.include_probe_vs_sft),
                "kmer_baseline_csv": str(PROJECT_ROOT / args.kmer_baseline_csv),
                "checkpoint_names": [spec["name"] for spec in specs],
                "missing_checkpoints": missing_checkpoints,
                "probe_vs_sft_command_count": sum(1 for row in commands if row["family"] == "probe_vs_sft"),
                "fresh_lora_command_count": sum(1 for row in commands if row["family"] == "fresh_lora"),
                "plan_path": str(plan_path),
            },
        ),
    )


def run_commands(commands: list[dict[str, Any]], cwd: Path) -> None:
    for item in commands:
        print(f"[stage1-formal] step={item['name']}", flush=True)
        result = subprocess.run(item["cmd"], cwd=str(cwd), check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=DEFAULT_PROJECT_PYTHON)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--formal-report-json", default=DEFAULT_FORMAL_REPORT)
    parser.add_argument("--option-b-best-candidate-json", default=DEFAULT_OPTION_B_BEST)
    parser.add_argument("--kmer-baseline-csv", default=DEFAULT_KMER_BASELINE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--checkpoint-mode",
        choices=["base_only", "all", "modified_only"],
        default="base_only",
        help="base_only runs the required calibration tranche before any modified checkpoint attacks.",
    )
    parser.add_argument(
        "--include-probe-vs-sft",
        action="store_true",
        help="Include the fresh-head probe tranche in addition to fresh_lora.",
    )
    parser.add_argument("--split-type", default="cluster_disjoint")
    parser.add_argument("--metric-for-best", default="auroc")
    parser.add_argument("--layers", default="3-9")
    parser.add_argument("--sft-layer", type=int, default=9)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--classification-head-lrs", default="1e-5,5e-5,1e-4")
    parser.add_argument("--lora-lrs", default="1e-5,5e-5,1e-4")
    parser.add_argument("--lora-ranks", default="8,16,32")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--probe-jobs", type=int, default=7)
    parser.add_argument("--feature-batch-size", type=int, default=0)
    parser.add_argument("--auto-batch-size", type=int, default=64)
    parser.add_argument("--classification-head-batch-size", type=int, default=2)
    parser.add_argument("--sft-steps", type=int, default=500)
    parser.add_argument("--sft-eval-every", type=int, default=50)
    parser.add_argument("--sft-patience", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lora-epochs", type=int, default=3)
    parser.add_argument("--lora-max-steps", type=int, default=0)
    parser.add_argument("--lora-eval-every", type=int, default=200)
    parser.add_argument("--validation-max-rows", type=int, default=0)
    parser.add_argument("--test-max-rows", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    args.classification_head_lrs = [float(part.strip()) for part in str(args.classification_head_lrs).split(",") if part.strip()]
    args.lora_lrs = [float(part.strip()) for part in str(args.lora_lrs).split(",") if part.strip()]
    args.lora_ranks = [int(part.strip()) for part in str(args.lora_ranks).split(",") if part.strip()]

    report_path = PROJECT_ROOT / args.formal_report_json
    best_candidate_json = PROJECT_ROOT / args.option_b_best_candidate_json
    kmer_baseline_csv = PROJECT_ROOT / args.kmer_baseline_csv
    tasks, blocked_tasks, merged_manifest = available_formal_tasks(report_path)
    include_modified = args.checkpoint_mode != "base_only"
    specs, missing_checkpoints = checkpoint_specs(best_candidate_json, include_modified=include_modified)
    if args.checkpoint_mode == "modified_only":
        specs = [spec for spec in specs if spec["name"] != "base"]
    baseline_map = load_kmer_baseline_map(kmer_baseline_csv) if kmer_baseline_csv.exists() else {}

    commands: list[dict[str, Any]] = []
    if args.include_probe_vs_sft:
        commands.extend(build_probe_vs_sft_commands(args, merged_manifest=merged_manifest, tasks=tasks, specs=specs))
    commands.extend(
        build_lora_commands(
            args,
            merged_manifest=merged_manifest,
            tasks=tasks,
            specs=specs,
            baseline_map=baseline_map,
        )
    )
    out_root = PROJECT_ROOT / args.out_root
    write_plan(
        out_root,
        args=args,
        commands=commands,
        tasks=tasks,
        blocked_tasks=blocked_tasks,
        specs=specs,
        missing_checkpoints=missing_checkpoints,
        merged_manifest=merged_manifest,
    )
    print(f"[stage1-formal] wrote {out_root / 'stage1_formal_experiment_plan.json'}")
    print(f"[stage1-formal] available tasks: {tasks}")
    if blocked_tasks:
        print(f"[stage1-formal] blocked tasks: {[row.get('task', '') for row in blocked_tasks]}")
    if missing_checkpoints:
        print(f"[stage1-formal] missing checkpoints skipped: {missing_checkpoints}")
    if not args.execute:
        return
    run_commands(commands, PROJECT_ROOT)


if __name__ == "__main__":
    main()
