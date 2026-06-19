import argparse
import glob
import os
from typing import Dict

import numpy as np
import pandas as pd


def load_layer_features(feature_dir: str, layer: int) -> np.ndarray:
    chunk_files = sorted(glob.glob(os.path.join(feature_dir, f"layer_{layer}", "chunk_*.npy")))
    if not chunk_files:
        raise FileNotFoundError(f"No feature chunks found for layer {layer}")
    return np.concatenate([np.load(path) for path in chunk_files], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose extracted layer-wise feature tensors.")
    parser.add_argument("--feature-dir", default="data/family_targets/coronaviridae/features")
    parser.add_argument("--out", default="data/family_targets/coronaviridae/features/feature_diagnostics.csv")
    parser.add_argument("--duplicate-atol", type=float, default=0.0)
    args = parser.parse_args()

    layer_dirs = sorted(glob.glob(os.path.join(args.feature_dir, "layer_*")))
    layers = sorted(int(os.path.basename(path).split("_")[-1]) for path in layer_dirs)
    if not layers:
        raise FileNotFoundError(f"No layer_* directories found in {args.feature_dir}")

    rows = []
    previous = None
    previous_layer = None
    for layer in layers:
        features = load_layer_features(args.feature_dir, layer)
        row: Dict[str, float | int | bool] = {
            "layer": layer,
            "n": features.shape[0],
            "dim": features.shape[1],
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "nan_count": int(np.isnan(features).sum()),
            "inf_count": int(np.isinf(features).sum()),
            "duplicate_previous": False,
            "max_abs_diff_previous": np.nan,
        }
        if previous is not None:
            max_diff = float(np.max(np.abs(features - previous)))
            row["max_abs_diff_previous"] = max_diff
            row["duplicate_previous"] = max_diff <= args.duplicate_atol
            if row["duplicate_previous"]:
                print(f"Layer {layer} is identical to layer {previous_layer}.")
        rows.append(row)
        previous = features
        previous_layer = layer

    diagnostics = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    diagnostics.to_csv(args.out, index=False)
    print(diagnostics.to_string(index=False))
    print(f"Wrote diagnostics to {args.out}")


if __name__ == "__main__":
    main()
