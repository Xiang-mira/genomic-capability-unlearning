"""Summarize the 15h route-decision package into machine-readable outputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BASE_RETAIN_PPL = 4.23097284833406
TARGET_TASKS = ["hvue_human_host_tropism", "hvue_human_transmissibility_coronaviridae"]
RETAIN_TASKS = [
    "gue_emp_h3",
    "gue_human_tf_1",
    "gue_mouse_1",
    "gue_prom_300_notata",
    "gue_splice_reconstructed",
]

CHECKPOINT_ORDER = [
    "base",
    "projection_rank32",
    "best_gd_from_task5a",
    "gd_random_control",
    "gd_loc_s1000",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def safe_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_mean(summary: dict[str, Any], group: str) -> float | None:
    return safe_float(((summary.get("groups") or {}).get(group) or {}).get("mean_score"))


def task_mean(summary: dict[str, Any], task: str) -> float | None:
    return safe_float(((summary.get("tasks") or {}).get(task) or {}).get("mean_score"))


def load_task5a_map(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    by_name = {row["checkpoint_name"]: row for row in rows if row.get("checkpoint_name")}
    alias = {
        "best_gd_from_task5a": by_name.get("gd_full_control", {}),
        "projection_rank32": by_name.get("projection_rank32", {}),
        "gd_random_control": by_name.get("gd_random_control", {}),
        "gd_loc_s1000": by_name.get("gd_loc_s1000", {}),
    }
    alias["base"] = {
        "checkpoint_name": "base",
        "method_family": "base",
        "retain_ppl": str(BASE_RETAIN_PPL),
        "retain_ppl_delta_vs_base": "0.0",
        "retain_safety_flag": "not_applicable",
    }
    return alias


def benchmark_path(root: Path, name: str) -> Path:
    return root / "benchmarks" / name / "eval_benchmarks_summary.json"


def existing_result_path(project_root: Path, name: str) -> Path:
    mapping = {
        "base": project_root / "data/phase2/base_benchmarks_slim/eval_benchmarks_summary.json",
        "gd_loc_s1000": project_root / "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/eval_benchmarks_summary.json",
    }
    return mapping[name]


def checkpoint_family(name: str) -> str:
    if name == "base":
        return "base"
    if name == "projection_rank32":
        return "projection"
    if name.startswith("gd_") or name == "best_gd_from_task5a":
        return "gd"
    return "other"


def classify_row(row: dict[str, Any], random_target_drop: float | None) -> dict[str, Any]:
    target_drop = row.get("target_drop_mean")
    retain_delta = row.get("retain_delta_mean")
    retain_flag = row.get("retain_flag")
    is_targeted_gd = row["checkpoint_name"] in {"best_gd_from_task5a", "gd_loc_s1000"}
    random_gap = None
    if is_targeted_gd and target_drop is not None and random_target_drop is not None:
        random_gap = target_drop - random_target_drop
    row["random_gap"] = random_gap

    if row["checkpoint_name"] == "base":
        row["absolute_signal_flag"] = "baseline_reference"
        row["decision"] = "baseline_reference"
        row["evidence_note"] = "base reference reused from slim benchmark"
        return row

    if row["checkpoint_name"] == "projection_rank32":
        row["absolute_signal_flag"] = "historical_reference"
        row["decision"] = "historical_reference"
        row["evidence_note"] = "projection remains a historical/reference anchor"
        return row

    if row["checkpoint_name"] == "gd_random_control":
        row["absolute_signal_flag"] = "random_control_reference"
        row["decision"] = "random_control_reference"
        row["evidence_note"] = "matched random control for GD specificity checks"
        return row

    if row["checkpoint_name"] == "best_gd_from_task5a" and retain_flag == "fail":
        row["absolute_signal_flag"] = "checkpoint_insufficient"
        row["decision"] = "stronger_but_unspecific_damage"
        row["evidence_note"] = "strong drop with retain failure; likely damage-heavy"
        return row

    if is_targeted_gd:
        relative_positive = bool(random_gap is not None and random_gap >= 0.03 and retain_delta is not None and retain_delta >= -0.03)
        route_seed = bool(relative_positive and target_drop is not None and target_drop >= 0.05 and retain_flag in {"pass", "warning"})
        if route_seed:
            row["absolute_signal_flag"] = "route_seed_candidate"
            row["decision"] = "promising_selective_signal"
            row["evidence_note"] = "targeted drop clears random gap and retain thresholds"
        elif relative_positive:
            row["absolute_signal_flag"] = "mechanism_hint_but_not_route_seed"
            row["decision"] = "mechanism_positive_checkpoint_insufficient"
            row["evidence_note"] = "targeted beats random, but absolute effect remains too weak"
        else:
            row["absolute_signal_flag"] = "route_not_worthy"
            row["decision"] = "likely_general_damage_or_non_specific"
            row["evidence_note"] = "targeted effect does not clearly outperform matched random"
        return row

    row["absolute_signal_flag"] = "unclassified_reference"
    row["decision"] = "reference_only"
    row["evidence_note"] = "reference-only checkpoint outside targeted-vs-random decision"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/home/teacher1/UT-project1/project1")
    parser.add_argument("--route-root", default="data/phase2/route_decision_20260715")
    parser.add_argument("--task5a-summary-csv", default="data/phase2/audits/task5a_identity_reaudit_20260713/task5a_identity_reaudit_summary.csv")
    parser.add_argument("--task5b-decision-md", default="data/phase2/audits/task5b_capability_reaudit_20260713/task5b_decision.md")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    route_root = (project_root / args.route_root).resolve()
    out_dir = route_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    task5a_map = load_task5a_map((project_root / args.task5a_summary_csv).resolve())

    base_summary = read_json(existing_result_path(project_root, "base"))
    base_target_mean = group_mean(base_summary, "hvue_forget")
    base_retain_mean = group_mean(base_summary, "gue_retain")

    rows: list[dict[str, Any]] = []
    for name in CHECKPOINT_ORDER:
        if name in {"base", "gd_loc_s1000"}:
            summary_path = existing_result_path(project_root, name)
            result_source = "existing"
        else:
            summary_path = benchmark_path(route_root, name)
            result_source = "route_decision_run"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing benchmark summary for {name}: {summary_path}")
        summary = read_json(summary_path)
        task5a = task5a_map.get(name, {})
        target_drop_mean = None if name == "base" else (base_target_mean - group_mean(summary, "hvue_forget"))
        retain_delta_mean = None if name == "base" else (group_mean(summary, "gue_retain") - base_retain_mean)
        row = {
            "checkpoint_name": name,
            "method_family": checkpoint_family(name),
            "result_source": result_source,
            "summary_path": str(summary_path),
            "target_drop_mean": target_drop_mean,
            "retain_delta_mean": retain_delta_mean,
            "retain_ppl": safe_float(task5a.get("retain_ppl")),
            "retain_ppl_delta_vs_base": safe_float(task5a.get("retain_ppl_delta_vs_base")),
            "retain_flag": task5a.get("retain_safety_flag", "unknown"),
            "target_task_host_drop": None if name == "base" else (task_mean(base_summary, TARGET_TASKS[0]) - task_mean(summary, TARGET_TASKS[0])),
            "target_task_coro_drop": None if name == "base" else (task_mean(base_summary, TARGET_TASKS[1]) - task_mean(summary, TARGET_TASKS[1])),
            "retain_tasks_mean": group_mean(summary, "gue_retain"),
            "forget_tasks_mean": group_mean(summary, "hvue_forget"),
        }
        rows.append(row)

    random_target_drop = next(row["target_drop_mean"] for row in rows if row["checkpoint_name"] == "gd_random_control")
    rows = [classify_row(row, random_target_drop) for row in rows]

    go_candidates = [
        row
        for row in rows
        if row["checkpoint_name"] in {"best_gd_from_task5a", "gd_loc_s1000"}
        and row["decision"] == "promising_selective_signal"
    ]
    if go_candidates:
        winner = max(go_candidates, key=lambda row: row["target_drop_mean"])
        final_decision = "A: route-seed positive"
        go_no_go = "go"
    else:
        mechanism_positive = any(row["decision"] == "mechanism_positive_checkpoint_insufficient" for row in rows)
        if mechanism_positive:
            final_decision = "B: mechanism-positive but checkpoint-insufficient"
        else:
            final_decision = "C: mechanism-negative but decision-useful"
        go_no_go = "no_go"
        winner = None

    csv_fields = [
        "checkpoint_name",
        "method_family",
        "result_source",
        "target_drop_mean",
        "retain_delta_mean",
        "retain_ppl",
        "retain_ppl_delta_vs_base",
        "random_gap",
        "absolute_signal_flag",
        "retain_flag",
        "decision",
        "evidence_note",
    ]
    csv_path = out_dir / "route_decision_main_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    md_lines = [
        "# Route Decision Main Comparison",
        "",
        f"- final_decision: `{final_decision}`",
        f"- go_no_go: `{go_no_go}`",
        f"- winning_targeted_gd: `{winner['checkpoint_name']}`" if winner else "- winning_targeted_gd: `none`",
        "",
        "| checkpoint | method | target_drop_mean | retain_delta_mean | retain_ppl | random_gap | absolute_signal_flag | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {checkpoint_name} | {method_family} | {target_drop_mean} | {retain_delta_mean} | {retain_ppl} | {random_gap} | {absolute_signal_flag} | {decision} |".format(
                checkpoint_name=row["checkpoint_name"],
                method_family=row["method_family"],
                target_drop_mean="-" if row["target_drop_mean"] is None else f"{row['target_drop_mean']:.6f}",
                retain_delta_mean="-" if row["retain_delta_mean"] is None else f"{row['retain_delta_mean']:.6f}",
                retain_ppl="-" if row["retain_ppl"] is None else f"{row['retain_ppl']:.6f}",
                random_gap="-" if row["random_gap"] is None else f"{row['random_gap']:.6f}",
                absolute_signal_flag=row["absolute_signal_flag"],
                decision=row["decision"],
            )
        )
    (out_dir / "route_decision_main_comparison.md").write_text("\n".join(md_lines) + "\n")

    one_pager = [
        "# Dual-Axis Route Decision",
        "",
        f"- final_decision: `{final_decision}`",
        f"- go_no_go: `{go_no_go}`",
        "",
        "- `gd_loc_s1000` vs `gd_random_control`: "
        + next(row["decision"] for row in rows if row["checkpoint_name"] == "gd_loc_s1000"),
        "- `gd_loc_s1000` absolute value: "
        + next(row["absolute_signal_flag"] for row in rows if row["checkpoint_name"] == "gd_loc_s1000"),
        "- `best_gd_from_task5a`: "
        + next(row["decision"] for row in rows if row["checkpoint_name"] == "best_gd_from_task5a"),
        "- `projection_rank32`: historical/reference anchor",
        "- `RMU`: not in primary comparison; keep as secondary safe reference only",
        "",
        "All conclusions remain `diagnostic / non-formal / route-selection evidence`.",
    ]
    (out_dir / "route_decision_one_pager.md").write_text("\n".join(one_pager) + "\n")

    write_json(
        out_dir / "route_decision_summary.json",
        {
            "final_decision": final_decision,
            "go_no_go": go_no_go,
            "winner": None if winner is None else winner["checkpoint_name"],
            "rows": rows,
            "base_retain_ppl": BASE_RETAIN_PPL,
            "target_tasks": TARGET_TASKS,
            "retain_tasks": RETAIN_TASKS,
        },
    )


if __name__ == "__main__":
    main()
