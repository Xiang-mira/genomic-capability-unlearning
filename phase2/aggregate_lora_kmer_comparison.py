"""Aggregate k-mer and LoRA benchmark results into one comparison table."""
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


METRICS = ("accuracy", "f1", "auroc", "auprc", "mse", "rmse", "r2", "pearson")
FIELDNAMES = [
    "Task",
    "Metric",
    "k-mer",
    "Original Evo",
    "GD",
    "RMU",
    "Attack Variant",
    "Delta GD",
    "Delta RMU",
]


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


def read_csv_metrics(path: Path, metrics: Iterable[str]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task") or row.get("Task")
            if not task:
                continue
            for metric in metrics:
                value = parse_float(row.get(metric) or row.get(metric.upper()) or row.get(metric.capitalize()))
                if value is not None:
                    out[(task, metric)] = value
    return out


def read_run_dir(path: Optional[str]) -> Dict[Tuple[str, str], float]:
    if not path:
        return {}
    return read_csv_metrics(Path(path) / "eval_benchmarks.csv", METRICS)


def parse_attack_specs(values: list[str]) -> dict[str, Dict[Tuple[str, str], float]]:
    attacks = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).name
        attacks[name] = read_run_dir(path)
    return attacks


def write_comparison(args) -> None:
    kmer = read_csv_metrics(Path(args.kmer_csv), METRICS) if args.kmer_csv else {}
    base = read_run_dir(args.base_dir)
    gd = read_run_dir(args.gd_dir)
    rmu = read_run_dir(args.rmu_dir)
    attacks = parse_attack_specs(args.attack_dirs or [])

    keys = set(kmer) | set(base) | set(gd) | set(rmu)
    for attack_metrics in attacks.values():
        keys |= set(attack_metrics)
    attack_items = sorted(attacks.items()) if attacks else [("", {})]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for task, metric in sorted(keys):
            base_value = base.get((task, metric))
            gd_value = gd.get((task, metric))
            rmu_value = rmu.get((task, metric))
            for attack_name, attack_metrics in attack_items:
                attack_value = attack_metrics.get((task, metric))
                writer.writerow(
                    {
                        "Task": task,
                        "Metric": metric,
                        "k-mer": fmt(kmer.get((task, metric))),
                        "Original Evo": fmt(base_value),
                        "GD": fmt(gd_value),
                        "RMU": fmt(rmu_value),
                        "Attack Variant": f"{attack_name}:{fmt(attack_value)}" if attack_name else "",
                        "Delta GD": fmt(gd_value - base_value if gd_value is not None and base_value is not None else None),
                        "Delta RMU": fmt(rmu_value - base_value if rmu_value is not None and base_value is not None else None),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmer-csv", default="")
    parser.add_argument("--base-dir", default="")
    parser.add_argument("--gd-dir", default="")
    parser.add_argument("--rmu-dir", default="")
    parser.add_argument("--attack-dirs", nargs="*", default=[])
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()
    write_comparison(args)
    print(f"[aggregate] wrote {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
