import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_layer_features(layer_dir: str) -> np.ndarray:
    chunk_files = sorted(glob.glob(os.path.join(layer_dir, "chunk_*.npy")))
    if not chunk_files:
        raise FileNotFoundError(f"No feature chunks found in {layer_dir}")
    chunks = [np.load(path) for path in chunk_files]
    return np.concatenate(chunks, axis=0)


def train_one_layer(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: List[float],
    max_iter: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    best_val = -1.0
    best_metrics = {}
    best_meta = {}

    scaler = StandardScaler()
    features_scaled = features.astype(np.float32, copy=False)
    features_scaled = scaler.fit_transform(features_scaled[train_mask])
    scaled_all = np.empty(features.shape, dtype=np.float32)
    scaled_all[train_mask] = features_scaled
    scaled_all[val_mask] = scaler.transform(features[val_mask].astype(np.float32, copy=False))
    scaled_all[test_mask] = scaler.transform(features[test_mask].astype(np.float32, copy=False))

    for c in c_grid:
        clf = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=max_iter,
            class_weight="balanced",
            n_jobs=None,
        )
        clf.fit(scaled_all[train_mask], labels[train_mask])

        metrics = {}
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            probs = clf.predict_proba(scaled_all[mask])[:, 1]
            preds = (probs >= 0.5).astype(np.int64)
            metrics[f"{name}_acc"] = accuracy_score(labels[mask], preds)
            metrics[f"{name}_auroc"] = roc_auc_score(labels[mask], probs)

        if metrics["val_auroc"] > best_val:
            best_val = metrics["val_auroc"]
            best_metrics = metrics
            best_meta = {
                "C": c,
                "coef": clf.coef_.copy(),
                "intercept": clf.intercept_.copy(),
                "scaler_mean": scaler.mean_.copy(),
                "scaler_scale": scaler.scale_.copy(),
            }

    return best_metrics, best_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train L2 logistic probes per layer.")
    parser.add_argument("--feature-dir", default="data/host_tropism/features")
    parser.add_argument("--out-dir", default="data/host_tropism/probes")
    parser.add_argument("--c-grid", default="0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    labels = np.load(os.path.join(args.feature_dir, "labels.npy"))
    splits = np.load(os.path.join(args.feature_dir, "splits.npy"))

    os.makedirs(args.out_dir, exist_ok=True)

    c_grid = [float(x) for x in args.c_grid.split(",")]

    layer_dirs = sorted(
        glob.glob(os.path.join(args.feature_dir, "layer_*")),
        key=lambda path: int(os.path.basename(path).split("_")[-1]),
    )
    summary_rows = []

    for layer_dir in layer_dirs:
        layer_name = os.path.basename(layer_dir)
        layer_idx = int(layer_name.split("_")[-1])
        features = load_layer_features(layer_dir)
        metrics, meta = train_one_layer(features, labels, splits, c_grid, args.max_iter)

        probe_path = os.path.join(args.out_dir, f"layer_{layer_idx}.npz")
        np.savez(
            probe_path,
            coef=meta["coef"],
            intercept=meta["intercept"],
            C=meta["C"],
            scaler_mean=meta["scaler_mean"],
            scaler_scale=meta["scaler_scale"],
        )

        row = {"layer": layer_idx, "C": meta["C"], **metrics}
        summary_rows.append(row)
        print(f"Layer {layer_idx}: val_auroc={metrics['val_auroc']:.4f} C={meta['C']}")

    summary = pd.DataFrame(summary_rows).sort_values("layer")
    summary_path = os.path.join(args.out_dir, "probe_metrics_by_layer.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote metrics to {summary_path}")


if __name__ == "__main__":
    main()
