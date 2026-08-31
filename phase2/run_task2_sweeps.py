"""
Run declarative Phase 2 Task 2 GD/RMU sweeps.

Each selected experiment runs:
  train -> eval_unlearn -> optional taxonomy-held-out eval -> optional HVUE/GUE eval
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


METHOD_SCRIPT = {
    # Classic gradient difference: -a_f * CE(forget) + a_r * CE(retain).
    "gd": "phase2/unlearn_gd.py",
    # Representation misdirection, multi-layer.
    "rmu": "phase2/unlearn_rmu.py",
    # Training-free joint probe null-space projection.
    "probe_nullspace": "phase2/project_probe_nullspace.py",
    # Probe-boundary training; --forget-objective {logit_zero, component_zero}.
    "probe_guided": "phase2/unlearn_probe.py",
    # Probe-guided representation training with separate trainable / loss layers.
    # This objective lived in unlearn_gd.py for part of the project's history;
    # see the HISTORY note in phase2/unlearn_probe_repr.py before comparing to
    # archived results.
    "probe_repr": "phase2/unlearn_probe_repr.py",
}


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def add_args(cmd: List[str], args: Dict[str, object]) -> None:
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(flag_name(key))
            continue
        cmd.extend([flag_name(key), str(value)])


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("[sweep]", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(cmd, check=True, env=env)


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"runs": {}}
    with path.open() as f:
        return json.load(f)


def update_run_status(progress_path: Path, run_name: str, **updates: object) -> None:
    progress = load_progress(progress_path)
    runs = progress.setdefault("runs", {})
    run = runs.setdefault(run_name, {})
    run.update(updates)
    run["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    progress["updated_at"] = run["updated_at"]
    write_progress(progress_path, progress)


def internal_eval_complete(ckpt_dir: Path) -> bool:
    return (
        (ckpt_dir / "eval_auroc.csv").exists()
        and (ckpt_dir / "eval_ppl.json").exists()
        and (ckpt_dir / "eval_representation.csv").exists()
    )


def taxonomy_eval_complete(args, run_name: str) -> bool:
    return (Path(args.taxonomy_out_root) / run_name / "taxonomy_heldout_summary.json").exists()


def benchmark_eval_complete(ckpt_dir: Path) -> bool:
    if not (ckpt_dir / "eval_benchmarks_summary.json").exists():
        return False
    progress_path = ckpt_dir / "eval_benchmarks_progress.json"
    if not progress_path.exists():
        return False
    try:
        with progress_path.open() as f:
            progress = json.load(f)
    except json.JSONDecodeError:
        return False
    completed = int(progress.get("completed_tasks") or 0)
    expected = int(progress.get("expected_tasks") or 0)
    return progress.get("status") == "complete" and expected > 0 and completed >= expected


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def selected_groups(config: dict, selectors: Iterable[str]) -> set[str]:
    aliases = config.get("aliases", {})
    groups = {experiment["group"] for experiment in config.get("experiments", [])}
    selected: set[str] = set()
    for selector in selectors:
        if selector in aliases:
            selected.update(aliases[selector])
        elif selector in groups:
            selected.add(selector)
        else:
            raise ValueError(
                f"Unknown sweep selector {selector!r}. "
                f"Known aliases={sorted(aliases)} groups={sorted(groups)}"
            )
    return selected


def selected_experiments(config: dict, selectors: List[str]) -> List[dict]:
    groups = selected_groups(config, selectors)
    return [experiment for experiment in config.get("experiments", []) if experiment["group"] in groups]


def train_and_eval(args, experiment: dict, progress_path: Path) -> None:
    method = experiment["method"]
    if method not in METHOD_SCRIPT:
        raise ValueError(f"Unsupported method {method!r} for run {experiment['name']}")

    run_name = experiment["name"]
    experiment_args = experiment.get("args", {})
    ckpt_dir = Path(args.out_dir) / run_name
    ckpt_path = ckpt_dir / "weights.safetensors"
    update_run_status(
        progress_path,
        run_name,
        group=experiment.get("group"),
        method=method,
        checkpoint_dir=str(ckpt_dir),
        status="started",
    )

    if (
        args.resume
        and args.delete_checkpoint_after_internal_eval
        and internal_eval_complete(ckpt_dir)
    ):
        print(f"[sweep] skip completed ephemeral run: {run_name}")
        update_run_status(
            progress_path,
            run_name,
            train="complete",
            internal_eval="complete",
            status="complete",
        )
        return

    if args.resume and ckpt_path.exists():
        print(f"[sweep] skip existing train: {run_name}")
        update_run_status(progress_path, run_name, train="skipped_existing")
    else:
        train_cmd = [
            sys.executable,
            METHOD_SCRIPT[method],
            "--out-dir",
            args.out_dir,
            "--run-name",
            run_name,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--save-steps",
            args.save_steps,
        ]
        add_args(train_cmd, experiment_args)
        update_run_status(progress_path, run_name, train="running", status="training")
        run_command(train_cmd, args.dry_run)
        update_run_status(progress_path, run_name, train="complete")

    if args.dry_run:
        update_run_status(progress_path, run_name, status="dry_run")
        return
    if not ckpt_path.exists():
        update_run_status(progress_path, run_name, status="failed", error=f"Missing checkpoint: {ckpt_path}")
        raise FileNotFoundError(f"Expected checkpoint was not written: {ckpt_path}")

    if args.run_internal_eval:
        if args.resume and internal_eval_complete(ckpt_dir):
            print(f"[sweep] skip existing internal eval: {run_name}")
            update_run_status(progress_path, run_name, internal_eval="skipped_existing")
        else:
            eval_cmd = [
                sys.executable,
                "phase2/eval_unlearn.py",
                "--ckpt",
                str(ckpt_path),
                "--internal-target-config",
                args.internal_target_config,
                "--device",
                args.device,
                "--batch-size",
                str(args.eval_batch_size),
                "--max-length",
                str(args.max_length),
                "--layers",
                args.internal_layers,
            ]
            forget_csv = experiment_args.get("forget_csv")
            retain_csv = experiment_args.get("retain_csv")
            localized_layers_path = experiment_args.get("localized_layers_path")
            if forget_csv:
                eval_cmd.extend(["--forget-csv", str(forget_csv)])
            if retain_csv:
                eval_cmd.extend(["--retain-csv", str(retain_csv)])
            if localized_layers_path:
                eval_cmd.extend(["--localized-layers-path", str(localized_layers_path)])
            update_run_status(progress_path, run_name, internal_eval="running", status="internal_eval")
            run_command(eval_cmd, False)
            update_run_status(progress_path, run_name, internal_eval="complete")
    else:
        update_run_status(progress_path, run_name, internal_eval="skipped_disabled")

    if args.run_taxonomy:
        if args.resume and taxonomy_eval_complete(args, run_name):
            print(f"[sweep] skip existing taxonomy eval: {run_name}")
            update_run_status(progress_path, run_name, taxonomy_eval="skipped_existing")
        else:
            tax_out = Path(args.taxonomy_out_root) / run_name
            tax_cmd = [
                sys.executable,
                "phase2/eval_taxonomy_heldout.py",
                "--ckpt",
                str(ckpt_path),
                "--dataset",
                args.taxonomy_dataset,
                "--manifest",
                args.taxonomy_manifest,
                "--cini-input",
                args.taxonomy_cini_input,
                "--group-key",
                args.taxonomy_group_key,
                "--out-dir",
                str(tax_out),
                "--device",
                args.device,
                "--layers",
                args.bench_layers,
                "--batch-size",
                str(args.bench_batch_size),
                "--auto-batch-size",
                str(args.bench_auto_batch_size),
                "--cpu-threads",
                str(args.bench_cpu_threads),
                "--probe-jobs",
                str(args.bench_probe_jobs),
                "--progress-every",
                str(args.bench_progress_every),
                "--max-length",
                str(args.max_length),
            ]
            update_run_status(progress_path, run_name, taxonomy_eval="running", status="taxonomy_eval")
            run_command(tax_cmd, False)
            update_run_status(progress_path, run_name, taxonomy_eval="complete")

    if args.run_benchmarks:
        if args.resume and benchmark_eval_complete(ckpt_dir):
            print(f"[sweep] skip existing benchmark eval: {run_name}")
            update_run_status(progress_path, run_name, benchmark_eval="skipped_existing")
        else:
            bench_cmd = [
                sys.executable,
                "phase2/eval_benchmarks.py",
                "--ckpt",
                str(ckpt_path),
                "--benchmark-manifest",
                args.benchmark_manifest,
                "--benchmark-scope",
                args.benchmark_scope,
                "--resume",
                "--device",
                args.device,
                "--batch-size",
                str(args.bench_batch_size),
                "--train-batch-size",
                str(args.bench_train_batch_size),
                "--eval-batch-size",
                str(args.bench_eval_batch_size),
                "--validation-max-rows",
                str(args.bench_validation_max_rows),
                "--cpu-threads",
                str(args.bench_cpu_threads),
                "--epochs",
                str(args.bench_epochs),
                "--max-steps",
                str(args.bench_max_steps),
                "--eval-every",
                str(args.bench_eval_every),
                "--patience",
                str(args.bench_patience),
                "--lr",
                str(args.bench_lr),
                "--weight-decay",
                str(args.bench_weight_decay),
                "--lora-rank",
                str(args.bench_lora_rank),
                "--lora-alpha",
                str(args.bench_lora_alpha),
                "--lora-dropout",
                str(args.bench_lora_dropout),
                "--metric-for-best",
                args.bench_metric_for_best,
                "--progress-every",
                str(args.bench_progress_every),
                "--max-length",
                str(args.max_length),
            ]
            if args.bench_task_filter:
                bench_cmd.extend(["--task-filter", args.bench_task_filter])
            if args.bench_discard_task_checkpoint:
                bench_cmd.append("--discard-task-checkpoint")
            update_run_status(progress_path, run_name, benchmark_eval="running", status="benchmark_eval")
            run_command(bench_cmd, False)
            update_run_status(progress_path, run_name, benchmark_eval="complete")
    if args.delete_checkpoint_after_internal_eval and args.run_internal_eval and ckpt_path.exists():
        ckpt_path.unlink()
        update_run_status(
            progress_path,
            run_name,
            checkpoint_deleted_after_internal_eval=True,
        )
        print(f"[sweep] deleted temporary checkpoint after evaluations: {ckpt_path}")
    update_run_status(progress_path, run_name, status="complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selectors", nargs="*", default=["all"], help="Sweep aliases or groups from the config.")
    parser.add_argument("--config", default="phase2/sweep_configs/task2_sweeps.json")
    parser.add_argument("--out-dir", default=os.environ.get("TUNED_ROOT", "data/phase2/checkpoints_tuned"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH", "2")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH", "4")))
    parser.add_argument(
        "--internal-layers",
        default=os.environ.get("INTERNAL_LAYERS", "5-9"),
        help="Layers evaluated by eval_unlearn.py for the merged selective-unlearning objective.",
    )
    parser.add_argument(
        "--internal-target-config",
        default=os.environ.get("INTERNAL_TARGET_CONFIG", "phase2/internal_eval_targets.json"),
        help="JSON config listing the internal probe targets evaluated by eval_unlearn.py.",
    )
    parser.add_argument(
        "--delete-checkpoint-after-internal-eval",
        action="store_true",
        help=(
            "Delete weights.safetensors after internal evaluation. Useful for full-model "
            "layer scans whose temporary checkpoints are very large; selected layers must "
            "be retrained later if their weights are needed."
        ),
    )
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LEN", "512")))
    parser.add_argument(
        "--save-steps",
        default=os.environ.get("SWEEP_SAVE_STEPS", ""),
        help=(
            "Comma-separated intermediate checkpoint steps passed to unlearn_gd/unlearn_rmu. "
            "Defaults to empty for sweeps to avoid filling disk; set SWEEP_SAVE_STEPS=100,200,500,1000 "
            "for trajectory runs."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("RESUME_SWEEP", "1") != "0",
        help="Skip train/eval stages whose output artifacts already exist.",
    )
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-taxonomy", action="store_true", default=os.environ.get("RUN_TAXONOMY", "0") == "1")
    parser.add_argument("--run-benchmarks", action="store_true", default=os.environ.get("RUN_BENCHMARKS", "0") == "1")
    parser.add_argument(
        "--run-internal-eval",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("RUN_INTERNAL_EVAL", "1") != "0",
        help="Run legacy internal probe/PPL eval after training. Disable for probe-free LoRA selection grids.",
    )
    parser.add_argument("--taxonomy-dataset", default=os.environ.get("TAXONOMY_DATASET", "host_tropism"))
    parser.add_argument("--taxonomy-group-key", default=os.environ.get("TAXONOMY_GROUP_KEY", "auto"))
    parser.add_argument("--taxonomy-manifest", default=os.environ.get("TAXONOMY_MANIFEST", "data/host_tropism/manifest.csv"))
    parser.add_argument("--taxonomy-cini-input", default=os.environ.get("TAXONOMY_CINI_INPUT", "data/benchmarks/hvue_gue_manifest.csv"))
    parser.add_argument("--taxonomy-out-root", default=os.environ.get("TAXONOMY_OUT_ROOT", "data/phase2/taxonomy_heldout"))
    parser.add_argument("--benchmark-manifest", default=os.environ.get("BENCHMARK_MANIFEST", "data/benchmarks/hvue_gue_manifest.csv"))
    parser.add_argument("--benchmark-scope", default=os.environ.get("BENCHMARK_SCOPE", "all"))
    parser.add_argument("--bench-task-filter", default=os.environ.get("BENCH_TASK_FILTER", ""))
    parser.add_argument("--bench-layers", default=os.environ.get("BENCH_LAYERS", "5-9"))
    parser.add_argument("--bench-batch-size", type=int, default=int(os.environ.get("BENCH_BATCH", "1")))
    parser.add_argument(
        "--bench-train-batch-size",
        type=int,
        default=int(os.environ.get("BENCH_TRAIN_BATCH", os.environ.get("BENCH_BATCH", "1"))),
    )
    parser.add_argument(
        "--bench-eval-batch-size",
        type=int,
        default=int(os.environ.get("BENCH_EVAL_BATCH", os.environ.get("BENCH_BATCH", "1"))),
    )
    parser.add_argument(
        "--bench-validation-max-rows",
        type=int,
        default=int(os.environ.get("BENCH_VALIDATION_MAX_ROWS", "0")),
        help="Cap validation rows used for benchmark early stopping; 0 keeps the full validation split.",
    )
    parser.add_argument("--bench-auto-batch-size", type=int, default=int(os.environ.get("BENCH_AUTO_BATCH", "96")))
    parser.add_argument("--bench-cpu-threads", type=int, default=int(os.environ.get("BENCH_CPU_THREADS", "16")))
    parser.add_argument("--bench-probe-jobs", type=int, default=int(os.environ.get("BENCH_PROBE_JOBS", "7")))
    parser.add_argument("--bench-progress-every", type=int, default=int(os.environ.get("BENCH_PROGRESS_EVERY", "1")))
    parser.add_argument("--bench-epochs", type=int, default=int(os.environ.get("BENCH_EPOCHS", "3")))
    parser.add_argument("--bench-max-steps", type=int, default=int(os.environ.get("BENCH_MAX_STEPS", "0")))
    parser.add_argument("--bench-eval-every", type=int, default=int(os.environ.get("BENCH_EVAL_EVERY", "100")))
    parser.add_argument("--bench-patience", type=int, default=int(os.environ.get("BENCH_PATIENCE", "3")))
    parser.add_argument("--bench-lr", type=float, default=float(os.environ.get("BENCH_LR", "1e-4")))
    parser.add_argument("--bench-weight-decay", type=float, default=float(os.environ.get("BENCH_WEIGHT_DECAY", "0.0")))
    parser.add_argument("--bench-lora-rank", type=int, default=int(os.environ.get("BENCH_LORA_RANK", "8")))
    parser.add_argument("--bench-lora-alpha", type=int, default=int(os.environ.get("BENCH_LORA_ALPHA", "16")))
    parser.add_argument("--bench-lora-dropout", type=float, default=float(os.environ.get("BENCH_LORA_DROPOUT", "0.0")))
    parser.add_argument(
        "--bench-discard-task-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete temporary per-task LoRA best.pt files after test metrics are written.",
    )
    parser.add_argument("--bench-metric-for-best", default=os.environ.get("BENCH_METRIC_FOR_BEST", "auto"))
    args = parser.parse_args()

    config = load_config(args.config)
    experiments = selected_experiments(config, args.selectors)
    train_defaults = dict(config.get("train_defaults", {}))
    if not experiments:
        raise RuntimeError(f"No experiments selected by {args.selectors}")
    progress_path = Path(args.progress_path or Path(args.out_dir) / "sweep_progress.json")
    print(f"[sweep] selected {len(experiments)} experiments from {args.config}")
    print(f"[sweep] resume={args.resume} progress={progress_path}")
    write_progress(
        progress_path,
        {
            **load_progress(progress_path),
            "config": args.config,
            "selectors": args.selectors,
            "selected_runs": [experiment["name"] for experiment in experiments],
            "resume": args.resume,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    for experiment in experiments:
        experiment = {
            **experiment,
            "args": {**train_defaults, **experiment.get("args", {})},
        }
        print("")
        print("=" * 72)
        print(f"[sweep] RUN {experiment['name']} ({experiment['group']}, {experiment['method']})")
        print("=" * 72)
        train_and_eval(args, experiment, progress_path)


if __name__ == "__main__":
    main()
