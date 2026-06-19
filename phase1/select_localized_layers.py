import argparse
import json
import os
from typing import List

import pandas as pd


def parse_layers(spec: str, available_layers: List[int] | None = None) -> List[int]:
    if spec == "all":
        if available_layers is None:
            raise ValueError("'all' requires available_layers.")
        return available_layers
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select localized layers from activation patching summary.")
    parser.add_argument(
        "--summary-csv",
        default="data/family_targets/coronaviridae/activation_patching/patching_layer_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="data/family_targets/coronaviridae/localized_layers.json",
    )
    parser.add_argument("--stable-layers", default="0-10")
    parser.add_argument("--min-abs-effect", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.25)
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    stable_layers = parse_layers(args.stable_layers, sorted(int(layer) for layer in df["layer"].unique()))
    effect_col = None
    for candidate in ("mean_abs_delta_target_prob", "mean_abs_delta_human_prob"):
        if candidate in df.columns:
            effect_col = candidate
            break
    if effect_col is None:
        raise ValueError(f"No supported effect column in {args.summary_csv}: {list(df.columns)}")

    stable = df[df["layer"].isin(stable_layers)].copy()
    if stable.empty:
        raise RuntimeError("No stable-layer rows found in patching summary.")

    max_effect = float(stable[effect_col].max())
    threshold = max(args.min_abs_effect, args.relative_threshold * max_effect)
    chosen = sorted(int(layer) for layer in stable.loc[stable[effect_col] >= threshold, "layer"].tolist())
    if not chosen:
        raise RuntimeError(
            f"No localized layers met threshold={threshold:.6f} in stable range {stable_layers}."
        )

    contiguous = list(range(min(chosen), max(chosen) + 1))
    top_row = stable.sort_values(effect_col, ascending=False).iloc[0]
    payload = {
        "layers": contiguous,
        "seed": 42,
        "source_csv": args.summary_csv,
        "stable_layers": stable_layers,
        "effect_column": effect_col,
        "max_effect": max_effect,
        "threshold": threshold,
        "selected_sparse_layers": chosen,
        "primary_target_layer": int(top_row["layer"]),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
