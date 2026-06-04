"""Write a short random-vs-taxonomy-held-out interpretation report."""
import argparse
import json
import os
from pathlib import Path
from typing import Optional


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def taxonomy_score(summary: dict) -> Optional[float]:
    value = summary.get("taxonomy_heldout", {}).get("mean_score")
    return float(value) if value is not None else None


def fmt(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def interpretation(random_auroc: Optional[float], heldout_auroc: Optional[float]) -> str:
    if random_auroc is None or heldout_auroc is None:
        return (
            "One or both evaluations did not produce AUROC values, so shortcut strength "
            "cannot be interpreted from these artifacts."
        )
    delta = random_auroc - heldout_auroc
    if delta >= 0.10:
        return (
            "Large AUROC degradation under taxonomy-held-out splitting. This is strong "
            "evidence that random-split performance is partly driven by taxonomy identity "
            "or close taxonomic correlates rather than a taxonomy-invariant host-tropism signal."
        )
    if delta >= 0.03:
        return (
            "Moderate AUROC degradation under taxonomy-held-out splitting. Taxonomy shortcuts "
            "likely contribute to performance, but the representation may retain some "
            "cross-taxonomy host-tropism signal."
        )
    if delta > -0.03:
        return (
            "Little AUROC change under taxonomy-held-out splitting. There is no strong "
            "evidence from this comparison that random-split performance is dominated by "
            "taxonomy identity."
        )
    return (
        "Taxonomy-held-out AUROC exceeded random-split AUROC. This can happen from split "
        "composition or sampling variance and should be checked against split summaries "
        "before drawing a biological conclusion."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-summary", required=True)
    parser.add_argument("--taxonomy-summary", required=True)
    parser.add_argument("--out-md", default="data/phase2/taxonomy_heldout/host_tropism_interpretation.md")
    parser.add_argument("--out-json", default="data/phase2/taxonomy_heldout/host_tropism_interpretation.json")
    args = parser.parse_args()

    random_summary = load_json(args.random_summary)
    taxonomy_summary = load_json(args.taxonomy_summary)
    random_auroc = taxonomy_score(random_summary)
    heldout_auroc = taxonomy_score(taxonomy_summary)
    delta = None if random_auroc is None or heldout_auroc is None else random_auroc - heldout_auroc
    text = interpretation(random_auroc, heldout_auroc)

    payload = {
        "random_split_auroc": random_auroc,
        "taxonomy_heldout_auroc": heldout_auroc,
        "auroc_delta_random_minus_taxonomy": delta,
        "interpretation": text,
        "random_summary": args.random_summary,
        "taxonomy_summary": args.taxonomy_summary,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    md = [
        "# Host Tropism Taxonomy Shortcut Check",
        "",
        f"- Random-split AUROC: `{fmt(random_auroc)}`",
        f"- Taxonomy-held-out AUROC: `{fmt(heldout_auroc)}`",
        f"- AUROC delta (random - taxonomy): `{fmt(delta)}`",
        "",
        "## Interpretation",
        "",
        text,
        "",
        "## Artifacts",
        "",
        f"- Random split summary: `{args.random_summary}`",
        f"- Taxonomy-held-out summary: `{args.taxonomy_summary}`",
    ]
    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[report] wrote {args.out_md}")
    print(f"[report] wrote {args.out_json}")
    print(f"[report] random={fmt(random_auroc)} taxonomy={fmt(heldout_auroc)} delta={fmt(delta)}")


if __name__ == "__main__":
    main()
