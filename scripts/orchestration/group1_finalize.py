#!/usr/bin/env python3
"""Aggregate Group 1 acceptance state from expected output files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "xiang_execution"
EXPECTED = [
    "results/geneb_capacity/geneb_cnn_capacity_sweep.csv",
    "results/geneb_capacity/geneb_cnn_dev_selection.csv",
    "results/geneb_capacity/geneb_probe_layer_sweep.csv",
    "results/geneb_capacity/geneb_probe_dev_selection.csv",
    "results/geneb_capacity/geneb_final_comparison.csv",
    "results/geneb_capacity/geneb_split_audit.csv",
    "results/geneb_capacity/geneb_statistics.csv",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [p for p in EXPECTED if not (ROOT / p).exists()]
    status = "PASS" if not missing else "BLOCKED"
    recommendation = "READY_FOR_GROUP2" if not missing else "BLOCKED_BY_ACCESS"
    payload = {
        "status": status,
        "recommendation": recommendation,
        "missing_expected_outputs": missing,
    }
    (OUT / "GROUP1_ACCEPTANCE.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Group 1 Acceptance", "", f"Status: `{status}`", "", f"Recommendation: `{recommendation}`", ""]
    if missing:
        lines += ["Missing expected outputs:", ""]
        lines += [f"- `{p}`" for p in missing]
    (OUT / "GROUP1_ACCEPTANCE.md").write_text("\n".join(lines) + "\n")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
