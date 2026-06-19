"""Aggregate HVUE LoRA benchmark outputs into Base/GD/RMU comparison tables."""
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


METRICS = ("accuracy", "f1", "auroc", "auprc", "mse", "rmse", "r2", "pearson")
FIELDNAMES = ["Task", "Metric", "Original Evo", "GD", "RMU", "Delta GD", "Delta RMU"]


def parse_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def read_metrics(run_dir: Path, metrics: Iterable[str]) -> Dict[Tuple[str, str], float]:
    path = run_dir / "eval_benchmarks.csv"
    rows: Dict[Tuple[str, str], float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"]
            for metric in metrics:
                value = parse_float(row.get(metric))
                if value is not None:
                    rows[(task, metric)] = value
    return rows


def write_comparison(base_dir: Path, gd_dir: Path, rmu_dir: Path, out_csv: Path) -> None:
    base = read_metrics(base_dir, METRICS)
    gd = read_metrics(gd_dir, METRICS)
    rmu = read_metrics(rmu_dir, METRICS)
    keys = sorted(set(base) | set(gd) | set(rmu))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for task, metric in keys:
            base_value = base.get((task, metric))
            gd_value = gd.get((task, metric))
            rmu_value = rmu.get((task, metric))
            writer.writerow(
                {
                    "Task": task,
                    "Metric": metric,
                    "Original Evo": fmt(base_value),
                    "GD": fmt(gd_value),
                    "RMU": fmt(rmu_value),
                    "Delta GD": fmt(gd_value - base_value if gd_value is not None and base_value is not None else None),
                    "Delta RMU": fmt(rmu_value - base_value if rmu_value is not None and base_value is not None else None),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--gd-dir", required=True)
    parser.add_argument("--rmu-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    write_comparison(
        base_dir=Path(args.base_dir),
        gd_dir=Path(args.gd_dir),
        rmu_dir=Path(args.rmu_dir),
        out_csv=Path(args.out_csv),
    )
    print(f"[aggregate] wrote {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
