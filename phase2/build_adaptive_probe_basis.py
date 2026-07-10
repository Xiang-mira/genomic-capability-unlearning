"""
Build adaptive multi-direction probe bases from cached layer features.

This is the CPU-only INLP-style step used before projection:
  1. Train a fresh linear probe on current residual features.
  2. Convert the scaled probe coefficient back to raw feature coordinates.
  3. Orthogonalize it against previous directions for that layer.
  4. Project the residual features away from the new direction.
  5. Stop when held-out separability is below the configured threshold.

The output basis files are consumed by phase2/project_probe_nullspace.py via
--basis-dir. Files are written as:
  <out-dir>/<target-name>/layer_<N>.npz
with a `basis` array shaped [hidden_dim, rank].
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.probe_utils import parse_layers


def load_layer_features(layer_dir: str) -> np.ndarray:
    chunk_files = sorted(glob.glob(os.path.join(layer_dir, "chunk_*.npy")))
    if not chunk_files:
        raise FileNotFoundError(f"No feature chunks found in {layer_dir}")
    return np.concatenate([np.load(path) for path in chunk_files], axis=0).astype(np.float32, copy=False)


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def separability(auroc: float) -> float:
    if np.isnan(auroc):
        return float("nan")
    return float(max(auroc, 1.0 - auroc))


def parse_c_grid(spec: str) -> List[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def split_metrics(clf: LogisticRegression, scaled: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    probs = clf.predict_proba(scaled[mask])[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    auroc = safe_auroc(labels[mask], probs)
    return {
        "acc": float(accuracy_score(labels[mask], preds)),
        "mcc": float(matthews_corrcoef(labels[mask], preds)),
        "auroc": auroc,
        "separability": separability(auroc),
    }


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: List[float],
    max_iter: int,
    seed: int,
) -> Tuple[dict, np.ndarray]:
    masks = {name: splits == name for name in ("train", "val", "test")}
    for split, mask in masks.items():
        if mask.sum() == 0:
            raise ValueError(f"Missing {split} split for adaptive basis training")
        if len(np.unique(labels[mask])) < 2:
            raise ValueError(f"Split {split} has only one class")

    scaler = StandardScaler()
    scaled = np.empty(features.shape, dtype=np.float32)
    scaled[masks["train"]] = scaler.fit_transform(features[masks["train"]])
    scaled[masks["val"]] = scaler.transform(features[masks["val"]])
    scaled[masks["test"]] = scaler.transform(features[masks["test"]])

    best = None
    best_val = -float("inf")
    for c_value in c_grid:
        clf = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=max_iter,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(scaled[masks["train"]], labels[masks["train"]])
        val = split_metrics(clf, scaled, labels, masks["val"])
        if not np.isnan(val["separability"]) and val["separability"] > best_val:
            best_val = val["separability"]
            best = (c_value, clf, scaler, scaled)

    if best is None:
        raise RuntimeError("Could not fit any adaptive probe candidate")

    c_value, clf, scaler, scaled = best
    raw_direction = clf.coef_.reshape(-1).astype(np.float32) / np.clip(
        scaler.scale_.astype(np.float32), 1e-12, None
    )
    metrics = {"C": float(c_value)}
    for split, mask in masks.items():
        values = split_metrics(clf, scaled, labels, mask)
        for key, value in values.items():
            metrics[f"{split}_{key}"] = value
    return metrics, raw_direction


def orthogonalize(candidate: np.ndarray, basis: List[np.ndarray], tol: float = 1e-6) -> np.ndarray | None:
    vector = candidate.astype(np.float32, copy=True)
    for existing in basis:
        vector = vector - float(vector @ existing) * existing
    norm = float(np.linalg.norm(vector))
    if norm <= tol:
        return None
    return vector / norm


def remove_direction(features: np.ndarray, unit: np.ndarray) -> np.ndarray:
    return features - (features @ unit).reshape(-1, 1) * unit.reshape(1, -1)


def build_layer_basis(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    args,
    layer_idx: int,
) -> Tuple[np.ndarray, List[dict], dict]:
    residual = features.astype(np.float32, copy=True)
    basis: List[np.ndarray] = []
    rows: List[dict] = []
    final_metrics = None
    c_grid = parse_c_grid(args.c_grid)

    for rank_idx in range(1, args.max_rank + 1):
        metrics, raw_direction = train_probe(
            residual,
            labels,
            splits,
            c_grid,
            args.max_iter,
            args.seed + layer_idx * 100 + rank_idx,
        )
        row = {
            "layer": layer_idx,
            "candidate_rank": rank_idx,
            "accepted": 0,
            **metrics,
        }
        rows.append(row)
        stop_score = metrics[args.stop_split + "_separability"]
        if stop_score <= args.stop_separability:
            row["stop_reason"] = "below_threshold"
            final_metrics = metrics
            break

        unit = orthogonalize(raw_direction, basis)
        if unit is None:
            row["stop_reason"] = "dependent_direction"
            final_metrics = metrics
            break
        basis.append(unit)
        row["accepted"] = 1
        row["accepted_rank"] = len(basis)
        row["direction_norm"] = float(np.linalg.norm(raw_direction))
        residual = remove_direction(residual, unit).astype(np.float32, copy=False)
    else:
        rows[-1]["stop_reason"] = "max_rank"

    if final_metrics is None:
        final_metrics, _raw_direction = train_probe(
            residual,
            labels,
            splits,
            c_grid,
            args.max_iter,
            args.seed + layer_idx * 100 + args.max_rank + 1,
        )
        rows.append(
            {
                "layer": layer_idx,
                "candidate_rank": args.max_rank + 1,
                "accepted": 0,
                "accepted_rank": len(basis),
                "stop_reason": "final_after_max_rank",
                **final_metrics,
            }
        )

    if basis:
        return np.stack(basis, axis=1).astype(np.float32), rows, final_metrics
    return np.zeros((features.shape[1], 0), dtype=np.float32), rows, final_metrics


def discover_layers(feature_dir: str, layers_spec: str) -> List[int]:
    if layers_spec:
        return parse_layers(layers_spec)
    layer_dirs = sorted(
        glob.glob(os.path.join(feature_dir, "layer_*")),
        key=lambda path: int(os.path.basename(path).split("_")[-1]),
    )
    return [int(os.path.basename(path).split("_")[-1]) for path in layer_dirs]


def write_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "layer",
        "candidate_rank",
        "accepted",
        "accepted_rank",
        "C",
        "train_acc",
        "train_mcc",
        "train_auroc",
        "train_separability",
        "val_acc",
        "val_mcc",
        "val_auroc",
        "val_separability",
        "test_acc",
        "test_mcc",
        "test_auroc",
        "test_separability",
        "direction_norm",
        "stop_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data/family_targets/coronaviridae/features")
    parser.add_argument("--out-dir", default="data/phase2/adaptive_probe_bases")
    parser.add_argument("--target-name", default="coronaviridae")
    parser.add_argument("--layers", default="", help="Optional layers/ranges, e.g. 0-12.")
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--stop-separability", type=float, default=0.55)
    parser.add_argument("--stop-split", choices=["val", "test"], default="val")
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = np.load(os.path.join(args.feature_dir, "labels.npy")).astype(np.int64)
    splits = np.load(os.path.join(args.feature_dir, "splits.npy")).astype(str)
    layers = discover_layers(args.feature_dir, args.layers)

    target_dir = Path(args.out_dir) / args.target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {
        "feature_dir": args.feature_dir,
        "out_dir": args.out_dir,
        "target_name": args.target_name,
        "layers": layers,
        "max_rank": args.max_rank,
        "stop_separability": args.stop_separability,
        "stop_split": args.stop_split,
        "layer_ranks": {},
        "final_metrics": {},
    }
    for layer_idx in layers:
        features = load_layer_features(os.path.join(args.feature_dir, f"layer_{layer_idx}"))
        basis, rows, final_metrics = build_layer_basis(features, labels, splits, args, layer_idx)
        out_path = target_dir / f"layer_{layer_idx}.npz"
        np.savez(
            out_path,
            basis=basis,
            target_name=args.target_name,
            layer=layer_idx,
            rank=basis.shape[1],
        )
        all_rows.extend(rows)
        summary["layer_ranks"][str(layer_idx)] = int(basis.shape[1])
        summary["final_metrics"][str(layer_idx)] = final_metrics
        print(
            f"[adaptive-basis] target={args.target_name} layer={layer_idx} "
            f"rank={basis.shape[1]} final_{args.stop_split}_sep="
            f"{final_metrics[args.stop_split + '_separability']:.4f} path={out_path}"
        )

    write_csv(target_dir / "adaptive_basis_metrics.csv", all_rows)
    with (target_dir / "adaptive_basis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[adaptive-basis] wrote {target_dir / 'adaptive_basis_metrics.csv'}")
    print(f"[adaptive-basis] wrote {target_dir / 'adaptive_basis_summary.json'}")


if __name__ == "__main__":
    main()
