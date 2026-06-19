"""Aggregate intermediate checkpoint trajectory metrics.

Expected layout:
  <checkpoint-root>/<run>/step_000100/{meta.json,eval_ppl.json,eval_auroc.csv,...}
  <checkpoint-root>/<run>/{meta.json,eval_ppl.json,eval_auroc.csv,...}

The script is intentionally tolerant of missing artifacts so it can be used while
long evaluations are still filling in.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


csv.field_size_limit(sys.maxsize)

METRIC_KEYS = ("auroc", "macro_auroc", "accuracy")


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def parse_step(path: Path, meta: dict) -> int:
    if "checkpoint_step" in meta:
        return int(meta["checkpoint_step"])
    if path.name.startswith("step_"):
        return int(path.name.split("_", 1)[1])
    return int(meta.get("steps") or 0)


def read_internal_auroc(path: Path, layers: set[int]) -> Optional[float]:
    if not path.exists():
        return None
    values = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                layer = int(row["layer"])
                value = float(row["test_auroc"])
            except (KeyError, TypeError, ValueError):
                continue
            if layer in layers:
                values.append(value)
    return mean(values) if values else None


def metric_from_row(row: dict) -> Optional[float]:
    for key in METRIC_KEYS:
        raw = row.get(key, "")
        if raw == "":
            continue
        value = float(raw)
        if not np.isnan(value):
            return value
    return None


def summarize_benchmarks(path: Path) -> tuple[dict[str, float], list[dict]]:
    if not path.exists():
        return {}, []
    by_group: dict[str, list[float]] = {}
    task_rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = metric_from_row(row)
            if value is None:
                continue
            group = row.get("group", "")
            by_group.setdefault(group, []).append(value)
            task_rows.append(
                {
                    "benchmark": row.get("benchmark", ""),
                    "task": row.get("task", ""),
                    "group": group,
                    "layer": row.get("layer", ""),
                    "metric": value,
                }
            )
    return {group: mean(values) for group, values in by_group.items() if values}, task_rows


def parse_layers(spec: str) -> set[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def iter_checkpoint_dirs(roots: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for run_dir in sorted(root_path.iterdir()):
            if not run_dir.is_dir():
                continue
            for step_dir in sorted(run_dir.glob("step_*")):
                if (step_dir / "meta.json").exists():
                    dirs.append(step_dir)
            if (run_dir / "meta.json").exists():
                dirs.append(run_dir)
    return dirs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def maybe_delta(base: Optional[float], value: Optional[float], *, drop: bool) -> Optional[float]:
    if base is None or value is None:
        return None
    return base - value if drop else value - base


def plot_trajectory(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in sorted({row["run"] for row in rows}):
        sub = sorted([row for row in rows if row["run"] == run], key=lambda r: r["step"])
        steps = [row["step"] for row in sub]
        axes[0].plot(steps, [row.get("hvue_forget_drop") for row in sub], marker="o", label=run)
        axes[0].plot(steps, [row.get("internal_auroc_drop") for row in sub], marker="x", linestyle="--")
        axes[1].plot(steps, [row.get("gue_retain_delta") for row in sub], marker="o", label=f"{run} GUE")
        axes[1].plot(steps, [row.get("viral_retain_delta") for row in sub], marker="x", linestyle="--", label=f"{run} Viro")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_title("Forget trajectory")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("drop vs base")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("Retain trajectory")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("delta vs base")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-roots", nargs="+", default=["data/phase2/checkpoints_tuned"])
    parser.add_argument("--base-benchmarks", default="data/phase2/base_benchmarks/eval_benchmarks_summary.json")
    parser.add_argument("--internal-base-auroc", type=float, default=0.844)
    parser.add_argument("--layers", default="3-9")
    parser.add_argument("--out-dir", default="data/phase2/virobench_diagnostics")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    base_summary = load_json(Path(args.base_benchmarks)) or {}
    base_groups = {
        group: payload.get("mean_score")
        for group, payload in base_summary.get("groups", {}).items()
    }

    metric_rows = []
    task_rows = []
    for ckpt_dir in iter_checkpoint_dirs(args.checkpoint_roots):
        meta = load_json(ckpt_dir / "meta.json") or {}
        run = meta.get("parent_run") or ckpt_dir.name
        step = parse_step(ckpt_dir, meta)
        ppl = load_json(ckpt_dir / "eval_ppl.json") or {}
        internal = read_internal_auroc(ckpt_dir / "eval_auroc.csv", layers)
        groups, benchmark_task_rows = summarize_benchmarks(ckpt_dir / "eval_benchmarks.csv")
        row = {
            "run": run,
            "checkpoint_dir": str(ckpt_dir),
            "method": meta.get("method", ""),
            "condition": meta.get("condition", ""),
            "step": step,
            "internal_auroc_3_9": internal,
            "internal_auroc_drop": maybe_delta(args.internal_base_auroc, internal, drop=True),
            "forget_val_loss": ppl.get("forget_val_loss"),
            "forget_val_perplexity": ppl.get("forget_val_perplexity"),
            "retain_val_loss": ppl.get("retain_val_loss"),
            "retain_val_perplexity": ppl.get("retain_val_perplexity"),
            "hvue_forget_mean": groups.get("hvue_forget"),
            "hvue_forget_drop": maybe_delta(base_groups.get("hvue_forget"), groups.get("hvue_forget"), drop=True),
            "gue_retain_mean": groups.get("gue_retain"),
            "gue_retain_delta": maybe_delta(base_groups.get("gue_retain"), groups.get("gue_retain"), drop=False),
            "viral_retain_mean": groups.get("viral_retain"),
            "viral_retain_delta": maybe_delta(base_groups.get("viral_retain"), groups.get("viral_retain"), drop=False),
        }
        metric_rows.append(row)
        for task_row in benchmark_task_rows:
            task_rows.append({"run": run, "step": step, "checkpoint_dir": str(ckpt_dir), **task_row})

    metric_rows.sort(key=lambda row: (row["run"], row["step"]))
    task_rows.sort(key=lambda row: (row["run"], row["step"], row["group"], row["task"], row["layer"]))
    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "trajectory_metrics.csv",
        metric_rows,
        [
            "run", "checkpoint_dir", "method", "condition", "step",
            "internal_auroc_3_9", "internal_auroc_drop",
            "forget_val_loss", "forget_val_perplexity",
            "retain_val_loss", "retain_val_perplexity",
            "hvue_forget_mean", "hvue_forget_drop",
            "gue_retain_mean", "gue_retain_delta",
            "viral_retain_mean", "viral_retain_delta",
        ],
    )
    write_csv(
        out_dir / "trajectory_taskwise_hvue_gue_virobench.csv",
        task_rows,
        ["run", "step", "checkpoint_dir", "benchmark", "task", "group", "layer", "metric"],
    )
    plot_trajectory(metric_rows, out_dir / "forget_vs_retain_trajectory.png")
    print(f"[trajectory] wrote {out_dir}")


if __name__ == "__main__":
    main()
