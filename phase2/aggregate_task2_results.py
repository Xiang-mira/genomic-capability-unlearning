"""
Aggregate Phase 2 Task 2 unlearning results and select best checkpoints.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


CSV_FIELDS = [
    "run",
    "checkpoint_dir",
    "method",
    "condition",
    "layers",
    "lr",
    "steps",
    "alpha_forget",
    "alpha_retain",
    "steer_coef",
    "target_direction",
    "target_layer",
    "internal_auroc_3_9",
    "internal_auroc_drop",
    "host_tropism_internal_auroc",
    "host_tropism_internal_drop",
    "coronaviridae_internal_auroc",
    "coronaviridae_internal_drop",
    "internal_gate_pass",
    "internal_min_drop",
    "internal_mean_drop",
    "forget_ppl",
    "retain_ppl",
    "retain_representation_mse",
    "retain_original_modified_cosine",
    "forget_representation_mse",
    "forget_original_modified_cosine",
    "hvue_forget_mean",
    "hvue_forget_drop",
    "gue_retain_mean",
    "gue_retain_delta",
    "viral_retain_mean",
    "viral_retain_delta",
    "taxonomy_mean",
    "taxonomy_drop",
    "forget_signal",
    "retain_penalty",
    "selection_score",
]


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def read_internal_auroc(path: Path, layers: set[int]) -> Optional[float]:
    if not path.exists():
        return None
    values = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                layer = int(row["layer"])
                value = float(row["test_auroc"])
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("target"):
                continue
            if layer in layers:
                values.append(value)
    return float(np.mean(values)) if values else None


def read_internal_targets(
    auroc_path: Path,
    ppl_summary: dict,
    layers: set[int],
) -> dict:
    payload = ppl_summary.get("internal_targets")
    if payload:
        return {str(name): metrics for name, metrics in payload.items()}

    if not auroc_path.exists():
        return {}

    grouped: Dict[str, List[float]] = {}
    with auroc_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get("target")
            if not target:
                continue
            try:
                layer = int(row["layer"])
                value = float(row["test_auroc_drop"])
            except (KeyError, TypeError, ValueError):
                continue
            if layer in layers:
                grouped.setdefault(target, []).append(value)
    return {
        target: {
            "localized_test_auroc_drop": float(np.mean(values)),
            "localized_test_mean_auroc": None,
            "base_localized_test_mean_auroc": None,
        }
        for target, values in grouped.items()
        if values
    }


def read_representation_metrics(path: Path, layers: set[int]) -> dict:
    grouped = {"forget": {"mse": [], "cosine": []}, "retain": {"mse": [], "cosine": []}}
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                layer = int(row["layer"])
                subset = row["subset"]
                split = row["split"]
                mse = float(row["representation_mse"])
                cosine = float(row["original_modified_cosine"])
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("target") and row.get("target") != "coronaviridae":
                continue
            if layer in layers and split == "test" and subset in grouped:
                grouped[subset]["mse"].append(mse)
                grouped[subset]["cosine"].append(cosine)
    result = {}
    for subset, metrics in grouped.items():
        if metrics["mse"]:
            result[f"{subset}_representation_mse"] = float(np.mean(metrics["mse"]))
            result[f"{subset}_original_modified_cosine"] = float(np.mean(metrics["cosine"]))
    return result


def group_mean(summary: Optional[dict], group: str) -> Optional[float]:
    if not summary:
        return None
    try:
        value = summary["groups"][group]["mean_score"]
    except KeyError:
        return None
    return float(value)


def taxonomy_mean(summary: Optional[dict]) -> Optional[float]:
    if not summary:
        return None
    tax = summary.get("taxonomy_heldout", {})
    value = tax.get("mean_score")
    if value is None:
        return None
    return float(value)


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


def as_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta_drop(base: Optional[float], score: Optional[float]) -> Optional[float]:
    if base is None or score is None:
        return None
    return float(base - score)


def delta(score: Optional[float], base: Optional[float]) -> Optional[float]:
    if base is None or score is None:
        return None
    return float(score - base)


def find_run_dirs(roots: Iterable[str]) -> List[Path]:
    dirs: List[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and (child / "meta.json").exists():
                dirs.append(child)
    return dirs


def row_for_run(args, run_dir: Path, layers: set[int], baselines: dict) -> dict:
    meta = load_json(run_dir / "meta.json") or {}
    ppl = load_json(run_dir / "eval_ppl.json") or {}
    benchmark_summary = load_json(run_dir / "eval_benchmarks_summary.json")
    tax_summary = load_json(Path(args.taxonomy_root) / run_dir.name / "taxonomy_heldout_summary.json")
    if tax_summary is None:
        tax_summary = load_json(run_dir / "taxonomy_heldout_summary.json")

    internal = read_internal_auroc(run_dir / "eval_auroc.csv", layers)
    internal_targets = read_internal_targets(run_dir / "eval_auroc.csv", ppl, layers)
    representation = read_representation_metrics(run_dir / "eval_representation.csv", layers)
    h_forget = group_mean(benchmark_summary, "hvue_forget")
    g_retain = group_mean(benchmark_summary, "gue_retain")
    v_retain = group_mean(benchmark_summary, "viral_retain")
    tax_mean = taxonomy_mean(tax_summary)

    host_tropism_internal = internal_targets.get("host_tropism", {}).get("localized_test_mean_auroc")
    coronaviridae_internal = internal_targets.get("coronaviridae", {}).get("localized_test_mean_auroc")
    host_tropism_drop = internal_targets.get("host_tropism", {}).get("localized_test_auroc_drop")
    coronaviridae_drop = internal_targets.get("coronaviridae", {}).get("localized_test_auroc_drop")
    internal_min_drop = as_float(ppl.get("internal_min_drop"))
    internal_mean_drop = as_float(ppl.get("internal_mean_drop"))
    internal_gate_pass = ppl.get("internal_gate_pass")
    internal_drop = internal_mean_drop
    if internal_drop is None:
        internal_drop = delta_drop(args.internal_base_auroc, internal)
    hvue_drop = delta_drop(baselines["hvue_forget"], h_forget)
    gue_delta = delta(g_retain, baselines["gue_retain"])
    viral_delta = delta(v_retain, baselines["viral_retain"])
    tax_drop = delta_drop(baselines["taxonomy"], tax_mean)

    forget_signal = next((value for value in (tax_drop, hvue_drop, internal_min_drop, internal_drop) if value is not None), None)
    consistency_checks = []
    if internal_gate_pass is not None:
        consistency_checks.append(bool(internal_gate_pass))
    if hvue_drop is not None and internal_min_drop is not None:
        consistency_checks.append((hvue_drop > 0) == (internal_min_drop > 0))
    if tax_drop is not None and internal_min_drop is not None:
        consistency_checks.append((tax_drop > 0) == (internal_min_drop > 0))
    retain_ppl = as_float(ppl.get("retain_val_perplexity"))
    ppl_base = args.base_retain_ppl
    ppl_delta = None if retain_ppl is None else (retain_ppl - ppl_base)
    hard_gate_checks = [
        host_tropism_drop is not None and host_tropism_drop > 0.0,
        coronaviridae_drop is not None and coronaviridae_drop > 0.0,
        host_tropism_drop is not None and host_tropism_drop >= args.internal_drop_threshold,
        coronaviridae_drop is not None and coronaviridae_drop >= args.internal_drop_threshold,
        gue_delta is not None and gue_delta >= args.min_gue_retain_delta,
        ppl_delta is not None and ppl_delta <= args.max_retain_ppl_increase,
    ]
    forget_gate_pass = (all(consistency_checks) if consistency_checks else True) and all(hard_gate_checks)
    gue_penalty = max(0.0, -(gue_delta or 0.0))
    ppl_penalty = max(0.0, (retain_ppl or ppl_base) - ppl_base) * args.ppl_penalty_weight
    retain_penalty = gue_penalty + ppl_penalty
    raw_selection_score = (forget_signal if forget_signal is not None else -1e9) - retain_penalty
    selection_score = raw_selection_score if forget_gate_pass else -1e9

    return {
        "run": run_dir.name,
        "checkpoint_dir": str(run_dir),
        "method": meta.get("method", ""),
        "condition": meta.get("condition", ""),
        "layers": "|".join(str(x) for x in meta.get("layers", [])),
        "lr": meta.get("lr", ""),
        "steps": meta.get("steps", ""),
        "alpha_forget": meta.get("alpha_forget", ""),
        "alpha_retain": meta.get("alpha_retain", ""),
        "steer_coef": meta.get("steer_coef", ""),
        "target_direction": meta.get("target_direction", ""),
        "target_layer": meta.get("target_layer", ""),
        "internal_auroc_3_9": internal,
        "internal_auroc_drop": internal_drop,
        "host_tropism_internal_auroc": host_tropism_internal,
        "host_tropism_internal_drop": host_tropism_drop,
        "coronaviridae_internal_auroc": coronaviridae_internal,
        "coronaviridae_internal_drop": coronaviridae_drop,
        "internal_gate_pass": internal_gate_pass,
        "internal_min_drop": internal_min_drop,
        "internal_mean_drop": internal_mean_drop,
        "forget_ppl": as_float(ppl.get("forget_val_perplexity")),
        "retain_ppl": retain_ppl,
        "retain_representation_mse": representation.get("retain_representation_mse"),
        "retain_original_modified_cosine": representation.get("retain_original_modified_cosine"),
        "forget_representation_mse": representation.get("forget_representation_mse"),
        "forget_original_modified_cosine": representation.get("forget_original_modified_cosine"),
        "hvue_forget_mean": h_forget,
        "hvue_forget_drop": hvue_drop,
        "gue_retain_mean": g_retain,
        "gue_retain_delta": gue_delta,
        "viral_retain_mean": v_retain,
        "viral_retain_delta": viral_delta,
        "taxonomy_mean": tax_mean,
        "taxonomy_drop": tax_drop,
        "forget_signal": forget_signal,
        "retain_penalty": retain_penalty,
        "selection_score": selection_score,
    }


def sort_key(row: dict) -> tuple:
    return (
        row.get("method", ""),
        row.get("condition", ""),
        -float(row.get("selection_score") or -1e9),
        row.get("run", ""),
    )


def select_best(rows: List[dict]) -> List[dict]:
    best: Dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("method", ""), row.get("condition", ""))
        if not key[0] or not key[1]:
            continue
        if key not in best or float(row["selection_score"]) > float(best[key]["selection_score"]):
            best[key] = row
    return sorted(best.values(), key=sort_key)


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def print_table(rows: List[dict]) -> None:
    print(
        f"{'run':<34} {'method':<20} {'cond':<10} "
        f"{'forget':>8} {'retain_pen':>10} {'score':>8} {'tax':>8} {'hvue':>8} "
        f"{'gueΔ':>8} {'viralΔ':>8} {'rppl':>8}"
    )
    print("-" * 130)
    for row in sorted(rows, key=lambda r: -float(r.get("selection_score") or -1e9)):
        print(
            f"{row['run']:<34} {row['method']:<20} {row['condition']:<10} "
            f"{fmt(row.get('forget_signal')):>8} {fmt(row.get('retain_penalty')):>10} "
            f"{fmt(row.get('selection_score')):>8} {fmt(row.get('taxonomy_drop')):>8} "
            f"{fmt(row.get('hvue_forget_drop')):>8} {fmt(row.get('gue_retain_delta')):>8} "
            f"{fmt(row.get('viral_retain_delta')):>8} {fmt(row.get('retain_ppl')):>8}"
        )


def fmt(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-roots", nargs="+", default=["data/phase2/checkpoints", "data/phase2/checkpoints_tuned"])
    parser.add_argument("--taxonomy-root", default="data/phase2/taxonomy_heldout")
    parser.add_argument("--base-benchmarks", default="data/phase2/base_benchmarks/eval_benchmarks_summary.json")
    parser.add_argument("--base-taxonomy", default="data/phase2/taxonomy_heldout/base/taxonomy_heldout_summary.json")
    parser.add_argument("--out-dir", default="data/phase2/results")
    parser.add_argument("--layers", default="5-9")
    parser.add_argument("--internal-base-auroc", type=float, default=0.844)
    parser.add_argument("--base-retain-ppl", type=float, default=4.2)
    parser.add_argument("--ppl-penalty-weight", type=float, default=0.01)
    parser.add_argument("--internal-drop-threshold", type=float, default=0.05)
    parser.add_argument("--min-gue-retain-delta", type=float, default=-0.02)
    parser.add_argument("--max-retain-ppl-increase", type=float, default=0.30)
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    base_bench = load_json(Path(args.base_benchmarks))
    base_tax = load_json(Path(args.base_taxonomy))
    baselines = {
        "hvue_forget": group_mean(base_bench, "hvue_forget"),
        "gue_retain": group_mean(base_bench, "gue_retain"),
        "viral_retain": group_mean(base_bench, "viral_retain"),
        "taxonomy": taxonomy_mean(base_tax),
    }

    rows = [
        row_for_run(args, run_dir, layers, baselines)
        for run_dir in find_run_dirs(args.ckpt_roots)
    ]
    rows = sorted(rows, key=sort_key)
    best_rows = select_best(rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "task2_runs.csv", rows)
    write_csv(out_dir / "task2_best_checkpoints.csv", best_rows)
    summary = {
        "baselines": baselines,
        "internal_base_auroc": args.internal_base_auroc,
        "base_retain_ppl": args.base_retain_ppl,
        "internal_drop_threshold": args.internal_drop_threshold,
        "min_gue_retain_delta": args.min_gue_retain_delta,
        "max_retain_ppl_increase": args.max_retain_ppl_increase,
        "n_runs": len(rows),
        "n_best": len(best_rows),
        "best_runs": [
            {field: row.get(field) for field in CSV_FIELDS}
            for row in best_rows
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "task2_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if args.print_table:
        print_table(rows)
    print(f"[aggregate] wrote {out_dir / 'task2_runs.csv'}")
    print(f"[aggregate] wrote {out_dir / 'task2_best_checkpoints.csv'}")
    print(f"[aggregate] wrote {out_dir / 'task2_summary.json'}")


if __name__ == "__main__":
    main()
