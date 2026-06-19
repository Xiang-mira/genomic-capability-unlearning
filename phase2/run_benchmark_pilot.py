"""
Run the benchmark pilot workflow and, after ranking, full eval for top methods.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


PROBE_SELECTED_LEGACY_CANDIDATES = {"gd_full_ar5", "rmu_full_sc200"}
LORA_RESULT_SENTINELS = {"problem_type", "best_checkpoint", "lora_rank", "trainable_params"}


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("[pilot]", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def ensure_lora_result_schema(out_dir: Path) -> None:
    result_path = out_dir / "eval_benchmarks.csv"
    if not result_path.exists():
        return
    with result_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    missing = sorted(LORA_RESULT_SENTINELS - set(header))
    if missing:
        raise RuntimeError(
            f"Existing results at {result_path} do not look like LoRA benchmark "
            f"outputs; missing columns: {', '.join(missing)}. Use a clean "
            "--pilot-root/--full-out-root or remove the legacy probe outputs."
        )


def eval_command(
    args,
    manifest: str,
    out_dir: Path,
    ckpt: Optional[Path] = None,
) -> List[str]:
    cmd = [
        args.python,
        "-u",
        "phase2/eval_benchmarks.py",
        "--benchmark-manifest",
        manifest,
        "--benchmark-scope",
        args.benchmark_scope,
        "--out-dir",
        str(out_dir),
        "--resume",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--cpu-threads",
        str(args.cpu_threads),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--metric-for-best",
        args.metric_for_best,
        "--max-length",
        str(args.max_length),
        "--seed",
        str(args.seed),
    ]
    if args.task_filter:
        cmd.extend(["--task-filter", args.task_filter])
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    return cmd


def ensure_pilot_manifest(args) -> None:
    out_manifest = Path(args.pilot_manifest)
    if out_manifest.exists() and not args.force_manifest:
        print(f"[pilot] using existing pilot manifest {out_manifest}")
        return
    cmd = [
        args.python,
        "phase2/subsample_benchmark_manifest.py",
        "--input-manifest",
        args.full_manifest,
        "--output-manifest",
        args.pilot_manifest,
        "--seed",
        str(args.seed),
        "--keep-all-task-rows",
        str(args.keep_all_task_rows),
        "--train-per-label",
        str(args.train_per_label),
        "--val-per-label",
        str(args.val_per_label),
        "--test-per-label",
        str(args.test_per_label),
    ]
    run_command(cmd, args.dry_run)


def run_pilot(args) -> None:
    ensure_pilot_manifest(args)
    pilot_root = Path(args.pilot_root)
    ensure_lora_result_schema(pilot_root / "base")
    run_command(
        eval_command(args, args.pilot_manifest, pilot_root / "base"),
        args.dry_run,
    )
    for run_name in args.candidates:
        ckpt = Path(args.ckpt_root) / run_name / "weights.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {ckpt}")
        ensure_lora_result_schema(pilot_root / run_name)
        run_command(
            eval_command(args, args.pilot_manifest, pilot_root / run_name, ckpt=ckpt),
            args.dry_run,
        )
    if not args.skip_rank:
        rankings_csv = pilot_root / "pilot_rankings.csv"
        rankings_json = pilot_root / "pilot_rankings.json"
        rank_cmd = [
            args.python,
            "phase2/rank_benchmark_pilot.py",
            "--pilot-root",
            str(pilot_root),
            "--out-csv",
            str(rankings_csv),
            "--out-json",
            str(rankings_json),
            "--top-k",
            str(args.top_k),
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--seed",
            str(args.seed),
            "--print-table",
        ]
        run_command(rank_cmd, args.dry_run)


def load_top_runs(rankings_path: Path, top_k: int) -> List[str]:
    with rankings_path.open() as f:
        payload = json.load(f)
    return [row["run"] for row in payload.get("top_runs", [])[:top_k]]


def run_full_top(args) -> None:
    rankings_path = Path(args.rankings_json)
    top_runs = load_top_runs(rankings_path, args.top_k)
    if not top_runs:
        raise RuntimeError(f"No top runs found in {rankings_path}")
    ensure_lora_result_schema(Path(args.full_base_out_dir))
    run_command(
        eval_command(args, args.full_manifest, Path(args.full_base_out_dir)),
        args.dry_run,
    )
    for run_name in top_runs:
        ckpt_dir = Path(args.ckpt_root) / run_name
        ckpt = ckpt_dir / "weights.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {ckpt}")
        out_dir = Path(args.full_out_root) / run_name if args.full_out_root else ckpt_dir
        ensure_lora_result_schema(out_dir)
        run_command(
            eval_command(args, args.full_manifest, out_dir, ckpt=ckpt),
            args.dry_run,
        )


def discover_candidates(ckpt_root: str) -> List[str]:
    root = Path(ckpt_root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "weights.safetensors").exists()
    )


def parse_candidates(args) -> List[str]:
    if args.discover_candidates:
        candidates = discover_candidates(args.ckpt_root)
    else:
        candidates = list(args.candidates or [])
    if not candidates and args.mode == "pilot":
        raise SystemExit(
            "No checkpoint candidates were provided. The old defaults "
            "gd_full_ar5/rmu_full_sc200 were probe-selected legacy candidates, "
            "so primary LoRA selection now requires --candidates ... or "
            "--discover-candidates."
        )
    legacy = sorted(set(candidates) & PROBE_SELECTED_LEGACY_CANDIDATES)
    if legacy:
        print(
            "[pilot] warning: candidate set includes probe-selected legacy "
            f"checkpoint(s): {', '.join(legacy)}",
            flush=True,
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "full-top"])
    parser.add_argument("--full-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--pilot-manifest", default="data/benchmarks/hvue_gue_pilot_manifest.csv")
    parser.add_argument(
        "--benchmark-scope",
        choices=["hvue", "all", "task"],
        default="all",
        help=(
            "Rows to evaluate with eval_benchmarks.py. Default all is used for "
            "LoRA checkpoint reranking so retain-task penalties participate."
        ),
    )
    parser.add_argument(
        "--task-filter",
        default="",
        help="Optional comma-separated task filter passed through to eval_benchmarks.py.",
    )
    parser.add_argument("--pilot-root", default="data/phase2/benchmark_pilot_lora")
    parser.add_argument("--ckpt-root", default="data/phase2/checkpoints_tuned")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument(
        "--discover-candidates",
        action="store_true",
        help=(
            "Evaluate every direct child of --ckpt-root that has weights.safetensors. "
            "Use this for LoRA-based checkpoint selection instead of legacy "
            "probe-selected defaults."
        ),
    )
    parser.add_argument("--full-base-out-dir", default="data/phase2/base_benchmarks_lora")
    parser.add_argument("--full-out-root", default="data/phase2/full_benchmarks_lora")
    parser.add_argument("--rankings-json", default="data/phase2/benchmark_pilot_lora/pilot_rankings.json")
    parser.add_argument("--python", default=os.environ.get("PHASE2_PYTHON", sys.executable))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-all-task-rows", type=int, default=6000)
    parser.add_argument("--train-per-label", type=int, default=2000)
    parser.add_argument("--val-per-label", type=int, default=500)
    parser.add_argument("--test-per-label", type=int, default=1500)
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--metric-for-best", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    args.candidates = parse_candidates(args)

    if args.mode == "pilot":
        run_pilot(args)
    else:
        run_full_top(args)


if __name__ == "__main__":
    main()
