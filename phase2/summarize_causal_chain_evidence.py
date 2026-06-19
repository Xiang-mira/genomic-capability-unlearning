"""Summarize whether Phase 2 artifacts support the causal-chain claims."""
import argparse
import csv
import json
from pathlib import Path
from typing import Optional


OBJECTIVES = [
    {
        "objective": "host_tropism_target_validity",
        "scientific_claim": "host-tropism is decodable from representations beyond shortcut-prone random splits",
    },
    {
        "objective": "probe_metric_validity",
        "scientific_claim": "frozen probe metrics predict downstream supervised fine-tuning behavior",
    },
    {
        "objective": "gd_rmu_tradeoff",
        "scientific_claim": "GD/RMU should be compared by knowledge removal versus collateral damage",
    },
    {
        "objective": "mechanistic_trajectories",
        "scientific_claim": "stepwise dynamics reveal when forgetting and retention damage emerge",
    },
]


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def csv_nonempty(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return len(rows) > 1


def controlled_split_status(root: Path) -> list[dict]:
    rows = []
    patterns = [
        "host_tropism_base/*/taxonomy_heldout_summary.json",
        "host_tropism_*_base/*/taxonomy_heldout_summary.json",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    for summary_path in sorted(set(paths)):
        summary = load_json(summary_path) or {}
        payload = summary.get("taxonomy_heldout", {})
        split_path = summary_path.with_name("controlled_split_manifest.csv")
        rows.append(
            {
                "split": summary_path.parent.name,
                "status": payload.get("status", summary.get("status", "unknown")),
                "score": payload.get("mean_score"),
                "n_task_layers": payload.get("n_task_layers"),
                "scientific_claim": payload.get("scientific_claim", ""),
                "confound_removed": payload.get("confound_removed", ""),
                "has_reusable_split_manifest": split_path.exists() and split_path.stat().st_size > 0,
                "summary_path": str(summary_path),
            }
        )
    return rows


def probe_status(root: Path) -> dict:
    result_path = root / "probe_vs_sft" / "probe_vs_sft_results.csv"
    corr_path = root / "probe_vs_sft" / "probe_sft_correlation.json"
    corr = load_json(corr_path) or {}
    return {
        "results_csv": str(result_path),
        "results_present": csv_nonempty(result_path),
        "correlation_json": str(corr_path),
        "correlation_present": bool(corr),
        "pearson": corr.get("pearson"),
        "spearman": corr.get("spearman"),
        "tasks": sorted((corr.get("by_task") or {}).keys()),
    }


def trajectory_status(root: Path) -> dict:
    traj_root = root / "trajectory"
    return {
        "trajectory_metrics_present": csv_nonempty(traj_root / "trajectory_metrics.csv"),
        "taskwise_metrics_present": csv_nonempty(traj_root / "trajectory_taskwise_hvue_gue_virobench.csv"),
        "gd_convergence_present": csv_nonempty(traj_root / "gd_convergence_diagnostics.csv"),
        "rmu_convergence_present": csv_nonempty(traj_root / "rmu_convergence_diagnostics.csv"),
    }


def recommendations(controlled: list[dict], probe: dict, trajectory: dict) -> list[str]:
    recs = []
    present_splits = {row["split"] for row in controlled}
    for expected in ("random", "taxonomy", "homology", "within_group"):
        if not any(split.startswith(expected) for split in present_splits):
            recs.append(f"Run target-{expected.replace('_', '-')} to strengthen host-tropism target validation.")
    if not probe["results_present"]:
        recs.append("Run probe-vs-sft after at least one controlled split manifest exists.")
    if probe["results_present"] and probe["pearson"] is None:
        recs.append("Inspect probe-vs-SFT results: correlation could not be estimated from available pairs.")
    if not trajectory["trajectory_metrics_present"]:
        recs.append("Run aggregate-trajectory after step checkpoint evaluations are present.")
    return recs


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Phase 2 Causal-Chain Evidence Status",
        "",
        "## Objectives",
        "",
    ]
    for objective in OBJECTIVES:
        lines.append(f"- `{objective['objective']}`: {objective['scientific_claim']}")
    lines.extend(["", "## Controlled Host-Tropism Splits", ""])
    if payload["controlled_splits"]:
        for row in payload["controlled_splits"]:
            lines.append(
                f"- `{row['split']}`: score={row['score']} status={row['status']} "
                f"confound={row['confound_removed']}"
            )
    else:
        lines.append("- No controlled split results found yet.")
    lines.extend(["", "## Probe-vs-SFT", ""])
    probe = payload["probe_vs_sft"]
    lines.append(f"- results present: `{probe['results_present']}`")
    lines.append(f"- Pearson: `{probe['pearson']}`")
    lines.append(f"- Spearman: `{probe['spearman']}`")
    lines.append(f"- tasks: `{', '.join(probe['tasks'])}`")
    lines.extend(["", "## Trajectory/Trade-Off", ""])
    for key, value in payload["trajectory"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recommended Next Runs", ""])
    if payload["recommendations"]:
        for rec in payload["recommendations"]:
            lines.append(f"- {rec}")
    else:
        lines.append("- Evidence package is structurally complete; inspect metric values for scientific interpretation.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/phase2/causal_chain")
    parser.add_argument("--out-json", default="data/phase2/causal_chain/causal_chain_evidence_status.json")
    parser.add_argument("--out-md", default="data/phase2/causal_chain/causal_chain_evidence_status.md")
    args = parser.parse_args()

    root = Path(args.root)
    controlled = controlled_split_status(root)
    probe = probe_status(root)
    trajectory = trajectory_status(root)
    payload = {
        "objectives": OBJECTIVES,
        "controlled_splits": controlled,
        "probe_vs_sft": probe,
        "trajectory": trajectory,
        "recommendations": recommendations(controlled, probe, trajectory),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(Path(args.out_md), payload)
    print(f"[causal-chain] wrote {out_json}")
    print(f"[causal-chain] wrote {args.out_md}")


if __name__ == "__main__":
    main()
