"""Merge multiple Stage 2 attacked-variant specs into one reusable compare spec."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata


def load_variants(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return [dict(item) for item in payload]


def merge_variant_rows(paths: list[str]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_variants(path):
            variant_id = str(row.get("variant_id") or "")
            if not variant_id:
                raise ValueError(f"{path} contains a row without variant_id")
            current = merged.get(variant_id)
            if current is None:
                current = dict(row)
                current["attacked_ckpt_by_recipe"] = dict(row.get("attacked_ckpt_by_recipe") or {})
                current["recipe_ids"] = [str(recipe_id) for recipe_id in (row.get("recipe_ids") or []) if str(recipe_id)]
                merged[variant_id] = current
                continue

            for key in ("initializer_label", "k0_ckpt", "attacked_ckpt", "readout_disruption_flag"):
                incoming = row.get(key, "")
                existing = current.get(key, "")
                if incoming and existing and incoming != existing:
                    raise ValueError(f"variant {variant_id} has conflicting {key}: {existing!r} vs {incoming!r}")
                if incoming and not existing:
                    current[key] = incoming

            attacked_map = dict(current.get("attacked_ckpt_by_recipe") or {})
            for recipe_id, ckpt_path in dict(row.get("attacked_ckpt_by_recipe") or {}).items():
                if recipe_id in attacked_map and attacked_map[recipe_id] != ckpt_path:
                    raise ValueError(
                        f"variant {variant_id} recipe {recipe_id} has conflicting attacked checkpoint paths"
                    )
                attacked_map[str(recipe_id)] = str(ckpt_path)
            current["attacked_ckpt_by_recipe"] = attacked_map

            recipe_ids = list(current.get("recipe_ids") or [])
            for recipe_id in row.get("recipe_ids") or []:
                recipe_id = str(recipe_id)
                if recipe_id and recipe_id not in recipe_ids:
                    recipe_ids.append(recipe_id)
            current["recipe_ids"] = recipe_ids

    rows = list(merged.values())
    rows.sort(key=lambda row: str(row.get("variant_id") or ""))
    for row in rows:
        attacked_map = dict(row.get("attacked_ckpt_by_recipe") or {})
        row["attacked_ckpt_by_recipe"] = {key: attacked_map[key] for key in sorted(attacked_map)}
        recipe_ids = [recipe_id for recipe_id in row.get("recipe_ids") or [] if recipe_id]
        row["recipe_ids"] = recipe_ids
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant-spec-json",
        action="append",
        dest="variant_specs",
        required=True,
        help="Repeatable Stage 2 attacked variant spec input.",
    )
    parser.add_argument(
        "--out-json",
        default="data/phase2/stage2_attacked_compare_merged_variants.json",
    )
    args = parser.parse_args()

    rows = merge_variant_rows(args.variant_specs)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    write_metadata(
        out_path.with_name(f"{out_path.stem}_metadata.json"),
        build_run_metadata(
            args=args,
            data_paths=args.variant_specs,
            extra={
                "phase": "merge_stage2_attacked_variants",
                "out_json": str(out_path),
                "variant_count": len(rows),
                "variant_ids": [str(row.get("variant_id") or "") for row in rows],
            },
        ),
    )
    print(f"[stage2-merge] wrote {out_path}")


if __name__ == "__main__":
    main()
