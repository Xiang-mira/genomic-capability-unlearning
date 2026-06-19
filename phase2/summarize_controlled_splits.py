"""Summarize host-tropism controlled split artifacts into one table."""
import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def score_from_summary(summary: Optional[dict]) -> Optional[float]:
    if not summary:
        return None
    if "taxonomy_heldout" in summary:
        return summary.get("taxonomy_heldout", {}).get("mean_score")
    groups = summary.get("groups", {})
    for group in ("hvue_forget", "primary_forget", "host_tropism"):
        if group in groups:
            return groups[group].get("mean_score")
    return summary.get("mean_score")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "split_type",
        "run",
        "summary_path",
        "score",
        "n_task_layers",
        "scientific_claim",
        "confound_removed",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_split_type(path: Path) -> str:
    text = str(path).lower()
    if "random" in text:
        return "random"
    if "within" in text:
        return "within_family"
    if "homology" in text or "cluster" in text:
        return "homology_aware"
    if "family" in text or "taxonomy" in text or "heldout" in text:
        return "family_or_taxonomy_heldout"
    return "unspecified"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-globs",
        nargs="+",
        default=[
            "data/phase2/taxonomy_heldout/*/*summary.json",
            "data/phase2/controlled_splits/*/*summary.json",
        ],
    )
    parser.add_argument("--out", default="data/phase2/virobench_diagnostics/host_tropism_controlled_split_results.csv")
    args = parser.parse_args()

    paths = []
    for pattern in args.summary_globs:
        paths.extend(sorted(Path().glob(pattern)))
    rows = []
    for path in paths:
        summary = load_json(path)
        if not summary:
            continue
        score = score_from_summary(summary)
        rows.append({
            "split_type": infer_split_type(path),
            "run": path.parent.name,
            "summary_path": str(path),
            "score": score,
            "n_task_layers": summary.get("taxonomy_heldout", {}).get("n_task_layers", ""),
            "scientific_claim": summary.get("taxonomy_heldout", {}).get("scientific_claim", ""),
            "confound_removed": summary.get("taxonomy_heldout", {}).get("confound_removed", ""),
            "notes": summary.get("notes", ""),
        })
    write_csv(Path(args.out), rows)
    print(f"[controlled-splits] wrote {args.out}")


if __name__ == "__main__":
    main()
