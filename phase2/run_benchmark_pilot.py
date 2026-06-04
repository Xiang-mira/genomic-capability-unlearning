"""
Run the benchmark pilot workflow and, after ranking, full eval for top methods.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_CANDIDATES = [
    "gd_full_ar5",
    "gd_localized_ar5_s1000",
    "rmu_full_sc200",
    "rmu_full_sc100",
]


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("[pilot]", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def eval_command(
    args,
    manifest: str,
    out_dir: Path,
    ckpt: Optional[Path] = None,
    feature_cache_dir: Optional[Path] = None,
) -> List[str]:
    cmd = [
        args.python,
        "-u",
        "phase2/eval_benchmarks.py",
        "--benchmark-manifest",
        manifest,
        "--out-dir",
        str(out_dir),
        "--resume",
        "--device",
        args.device,
        "--layers",
        args.layers,
        "--batch-size",
        str(args.batch_size),
        "--auto-batch-size",
        str(args.auto_batch_size),
        "--cpu-threads",
        str(args.cpu_threads),
        "--probe-jobs",
        str(args.probe_jobs),
        "--progress-every",
        str(args.progress_every),
        "--max-length",
        str(args.max_length),
    ]
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    if feature_cache_dir is not None:
        cmd.extend(["--feature-cache-dir", str(feature_cache_dir)])
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
    cache_dir = Path(args.feature_cache_dir or pilot_root / "feature_cache")
    run_command(
        eval_command(args, args.pilot_manifest, pilot_root / "base", feature_cache_dir=cache_dir),
        args.dry_run,
    )
    for run_name in args.candidates:
        ckpt = Path(args.ckpt_root) / run_name / "weights.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {ckpt}")
        run_command(
            eval_command(args, args.pilot_manifest, pilot_root / run_name, ckpt=ckpt, feature_cache_dir=cache_dir),
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
    cache_dir = Path(args.full_feature_cache_dir or "data/phase2/feature_cache")
    run_command(
        eval_command(args, args.full_manifest, Path(args.full_base_out_dir), feature_cache_dir=cache_dir),
        args.dry_run,
    )
    for run_name in top_runs:
        ckpt_dir = Path(args.ckpt_root) / run_name
        ckpt = ckpt_dir / "weights.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {ckpt}")
        out_dir = Path(args.full_out_root) / run_name if args.full_out_root else ckpt_dir
        run_command(
            eval_command(args, args.full_manifest, out_dir, ckpt=ckpt, feature_cache_dir=cache_dir),
            args.dry_run,
        )


def parse_candidates(values: Optional[Iterable[str]]) -> List[str]:
    return list(values) if values else list(DEFAULT_CANDIDATES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "full-top"])
    parser.add_argument("--full-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--pilot-manifest", default="data/benchmarks/hvue_gue_pilot_manifest.csv")
    parser.add_argument("--pilot-root", default="data/phase2/benchmark_pilot")
    parser.add_argument("--ckpt-root", default="data/phase2/checkpoints_tuned")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--full-feature-cache-dir", default=None)
    parser.add_argument("--full-base-out-dir", default="data/phase2/base_benchmarks")
    parser.add_argument("--full-out-root", default=None)
    parser.add_argument("--rankings-json", default="data/phase2/benchmark_pilot/pilot_rankings.json")
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
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--auto-batch-size", type=int, default=96)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--probe-jobs", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layers", default="3-9")
    args = parser.parse_args()
    args.candidates = parse_candidates(args.candidates)

    if args.mode == "pilot":
        run_pilot(args)
    else:
        run_full_top(args)


if __name__ == "__main__":
    main()
