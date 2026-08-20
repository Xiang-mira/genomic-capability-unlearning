"""
Shortcut-vs-model-probe audit for a single HVUE task.

Computes both AUROC and MCC for:
  1. Raw sequence features (GC, mononucleotide, dinucleotide)
  2. k-mer hash features (3–6 mer, configurable)
  3. Linear probe (logistic regression) on Evo base model hidden states,
     extracted via forward-hook mean-pooling at specified layer indices.

The three comparisons reveal how much of the task performance is explained
by sequence statistics alone vs. learned model representations, and whether
the model adds signal beyond what k-mers already provide.

Usage (two tasks in parallel, GPUs 6 and 7):
  CUDA_VISIBLE_DEVICES=6 python phase2/shortcut_vs_model_probe.py \\
    --manifest data/shortcut_audit/host_tropism_manifest.csv \\
    --task-name host_tropism --out-dir data/shortcut_audit/host_tropism \\
    --device cuda:0 --layers 0,5,9,12,15,20,25,31 &

  CUDA_VISIBLE_DEVICES=7 python phase2/shortcut_vs_model_probe.py \\
    --manifest data/shortcut_audit/cini_manifest.csv \\
    --task-name cini --out-dir data/shortcut_audit/cini \\
    --device cuda:0 --layers 0,5,9,12,15,20,25,31 &

  wait
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

csv.field_size_limit(sys.maxsize)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUCLEOTIDES = "ACGT"


# ---------------------------------------------------------------------------
# Feature builders (CPU, no GPU)
# ---------------------------------------------------------------------------

def sequence_stats_matrix(sequences: List[str]) -> np.ndarray:
    """GC + mono + dinucleotide frequencies → dense float32 matrix."""
    rows = []
    for seq in sequences:
        seq = (seq or "").upper()
        n = max(len(seq), 1)
        from collections import Counter as C
        cnt = C(seq)
        row = [
            len(seq) / 1000.0,
            (cnt["G"] + cnt["C"]) / n,
        ]
        for b in NUCLEOTIDES + "N":
            row.append(cnt[b] / n)
        two_n = max(len(seq) - 1, 1)
        two_cnt = C(seq[i:i+2] for i in range(max(len(seq)-1, 0)))
        for a in NUCLEOTIDES:
            for b in NUCLEOTIDES:
                row.append(two_cnt.get(a+b, 0) / two_n)
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def kmer_matrix(sequences: List[str], kmer_min: int = 3, kmer_max: int = 6) -> csr_matrix:
    def kmer_analyzer(seq):
        seq = seq.upper()
        for k in range(kmer_min, kmer_max + 1):
            for i in range(len(seq) - k + 1):
                yield seq[i:i+k]
    vec = HashingVectorizer(
        analyzer=kmer_analyzer,
        n_features=2**18,
        norm="l2",
        alternate_sign=False,
        dtype=np.float32,
    )
    return vec.fit_transform(sequences)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        return float("nan")


def separability(auroc: float) -> float:
    return max(auroc, 1.0 - auroc)


def fit_probe(
    X_train: csr_matrix | np.ndarray,
    y_train: np.ndarray,
    X_test: csr_matrix | np.ndarray,
    y_test: np.ndarray,
    c_grid: List[float],
    seed: int = 42,
    scale: bool = False,
) -> dict:
    """Fit logistic regression with C cross-validation on train, evaluate on test.

    Returns dict with train_auroc, test_auroc, test_mcc, test_sep, best_C.
    """
    if scale:
        if hasattr(X_train, "toarray"):
            scaler = StandardScaler(with_mean=False)
        else:
            scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    best_C, best_val_auroc = c_grid[0], -1.0
    # simple hold-out from train: last 20% as val
    n_val = max(1, len(y_train) // 5)
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]

    if len(np.unique(y_tr)) < 2:
        return {"train_auroc": float("nan"), "test_auroc": float("nan"),
                "test_mcc": float("nan"), "test_sep": float("nan"), "best_C": c_grid[0]}

    for C in c_grid:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=seed, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        auc = safe_auroc(y_val, proba)
        if auc > best_val_auroc:
            best_val_auroc, best_C = auc, C

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs", random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)

    train_proba = clf.predict_proba(X_train)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_pred = clf.predict(X_test)

    return {
        "train_auroc": safe_auroc(y_train, train_proba),
        "test_auroc": safe_auroc(y_test, test_proba),
        "test_mcc": safe_mcc(y_test, test_pred),
        "test_sep": separability(safe_auroc(y_test, test_proba)),
        "best_C": best_C,
    }


# ---------------------------------------------------------------------------
# Model feature extraction (GPU)
# ---------------------------------------------------------------------------

def load_evo(model_dir: str, config_path: str, device: str):
    from phase1.utils import load_local_checkpoint
    model = load_local_checkpoint(model_dir, config_path, device=device)
    model.eval()
    return model


def extract_model_features(
    model,
    sequences: List[str],
    layers: List[int],
    batch_size: int,
    max_length: int,
    device: str,
) -> Dict[int, np.ndarray]:
    import torch
    from phase2.utils import tokenize_batch
    from evo.tokenizer import CharLevelTokenizer
    tokenizer = CharLevelTokenizer(max_length)

    num_layers = len(model.blocks)
    buffers: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    state = {"mask": None}
    layers_set = set(layers)

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            if layer_idx not in layers_set:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx + 1 < num_layers:
                hidden = model.blocks[layer_idx + 1].pre_norm(hidden)
            else:
                hidden = model.norm(hidden)
            mask = state["mask"]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            buffers[layer_idx].append(pooled.detach().float().cpu().numpy())
        return hook

    handles = [model.blocks[l].register_forward_hook(make_hook(l)) for l in layers]
    n = len(sequences)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sequences[start:start + batch_size]
            ids, mask = tokenize_batch(batch, tokenizer, max_length, device)
            state["mask"] = mask
            _ = model(ids, padding_mask=mask)
            if (start // batch_size) % 50 == 0:
                elapsed = time.time() - t0
                pct = (start + len(batch)) / n * 100
                print(f"  [extract] {start+len(batch)}/{n} ({pct:.0f}%) {elapsed:.0f}s", flush=True)
    for h in handles:
        h.remove()
    return {l: np.concatenate(buffers[l], axis=0) for l in layers}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    rows = []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    seqs = [r["sequence"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    splits = np.array([r["split"] for r in rows])
    return seqs, labels, splits


def subsample_balanced(seqs, labels, splits, split_name, max_per_class, rng):
    """Return balanced subsample of a split, capped at max_per_class per class."""
    idx = np.where(splits == split_name)[0]
    if len(idx) == 0:
        return idx
    classes = np.unique(labels[idx])
    chosen = []
    for c in classes:
        cidx = idx[labels[idx] == c]
        if len(cidx) > max_per_class:
            cidx = rng.choice(cidx, size=max_per_class, replace=False)
        chosen.append(cidx)
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    return chosen


def print_row(name: str, result: dict, n_train: int, n_test: int):
    auroc = result.get("test_auroc", float("nan"))
    sep = result.get("test_sep", float("nan"))
    mcc = result.get("test_mcc", float("nan"))
    print(f"  {name:<40s}  n_train={n_train:>6d}  n_test={n_test:>6d}  "
          f"AUROC={auroc:.4f}  sep={sep:.4f}  MCC={mcc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--layers", default="0,5,9,12,15,20,25,31")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-per-class", type=int, default=5000,
                        help="Subsample train to this many per class (0=no limit).")
    parser.add_argument("--max-test-per-class", type=int, default=3000,
                        help="Subsample test to this many per class (0=no limit).")
    parser.add_argument("--kmer-min", type=int, default=3)
    parser.add_argument("--kmer-max", type=int, default=6)
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1.0,10.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model", action="store_true",
                        help="Only run sequence-feature baselines, no GPU.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*70}")
    print(f"Shortcut vs Model Probe Audit: {args.task_name}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Load manifest and build splits
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading manifest: {args.manifest}")
    seqs, labels, splits = load_manifest(args.manifest)
    print(f"  total={len(seqs)}  splits={dict(Counter(splits))}  labels={dict(Counter(labels))}")

    # Identify train/test split names
    split_names = set(splits)
    train_split = "train" if "train" in split_names else sorted(split_names)[0]
    test_split = "test" if "test" in split_names else (
        "val" if "val" in split_names else sorted(split_names)[-1]
    )
    print(f"  using train_split={train_split!r}  test_split={test_split!r}")

    train_idx = subsample_balanced(seqs, labels, splits, train_split,
                                   args.max_train_per_class if args.max_train_per_class > 0 else 10**9, rng)
    test_idx = subsample_balanced(seqs, labels, splits, test_split,
                                  args.max_test_per_class if args.max_test_per_class > 0 else 10**9, rng)

    seqs_tr = [seqs[i] for i in train_idx]
    seqs_te = [seqs[i] for i in test_idx]
    y_tr = labels[train_idx]
    y_te = labels[test_idx]
    print(f"  train: n={len(seqs_tr)} labels={dict(Counter(y_tr.tolist()))}")
    print(f"  test:  n={len(seqs_te)} labels={dict(Counter(y_te.tolist()))}")

    results = []

    # ------------------------------------------------------------------
    # Sequence-statistics baselines (CPU)
    # ------------------------------------------------------------------
    print(f"\n[2/4] Sequence-statistic baselines (CPU)...")

    t0 = time.time()
    X_raw_tr = csr_matrix(sequence_stats_matrix(seqs_tr))
    X_raw_te = csr_matrix(sequence_stats_matrix(seqs_te))
    r_raw = fit_probe(X_raw_tr, y_tr, X_raw_te, y_te, c_grid, seed=args.seed, scale=True)
    r_raw["feature"] = "raw_gc_mono_di"
    r_raw["n_train"] = len(seqs_tr)
    r_raw["n_test"] = len(seqs_te)
    results.append(r_raw)
    print_row("raw_gc_mono_di", r_raw, len(seqs_tr), len(seqs_te))
    print(f"  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  building k-mer features...", flush=True)
    X_kmer_tr = kmer_matrix(seqs_tr, args.kmer_min, args.kmer_max)
    X_kmer_te = kmer_matrix(seqs_te, args.kmer_min, args.kmer_max)
    r_kmer = fit_probe(X_kmer_tr, y_tr, X_kmer_te, y_te, c_grid, seed=args.seed)
    r_kmer["feature"] = f"kmer_{args.kmer_min}_{args.kmer_max}"
    r_kmer["n_train"] = len(seqs_tr)
    r_kmer["n_test"] = len(seqs_te)
    results.append(r_kmer)
    print_row(f"kmer_{args.kmer_min}_{args.kmer_max}", r_kmer, len(seqs_tr), len(seqs_te))
    print(f"  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    X_both_tr = hstack([X_raw_tr, X_kmer_tr], format="csr")
    X_both_te = hstack([X_raw_te, X_kmer_te], format="csr")
    r_both = fit_probe(X_both_tr, y_tr, X_both_te, y_te, c_grid, seed=args.seed)
    r_both["feature"] = "raw_plus_kmer"
    r_both["n_train"] = len(seqs_tr)
    r_both["n_test"] = len(seqs_te)
    results.append(r_both)
    print_row("raw_plus_kmer", r_both, len(seqs_tr), len(seqs_te))
    print(f"  ({time.time()-t0:.1f}s)")

    best_shortcut_auroc = max(r["test_auroc"] for r in results if not np.isnan(r.get("test_auroc", float("nan"))))
    best_shortcut_sep = max(r["test_sep"] for r in results if not np.isnan(r.get("test_sep", float("nan"))))
    best_shortcut_mcc = max(r["test_mcc"] for r in results if not np.isnan(r.get("test_mcc", float("nan"))))

    # ------------------------------------------------------------------
    # Model probe (GPU)
    # ------------------------------------------------------------------
    if not args.skip_model:
        print(f"\n[3/4] Loading Evo base model on {args.device}...")
        import torch
        t0 = time.time()
        model = load_evo(args.model_dir, args.config_path, args.device)
        print(f"  loaded in {time.time()-t0:.1f}s")

        all_seqs = seqs_tr + seqs_te
        n_tr = len(seqs_tr)

        print(f"  extracting hidden states at layers {layers} for {len(all_seqs)} sequences...")
        t0 = time.time()
        features_all = extract_model_features(
            model, all_seqs, layers, args.batch_size, args.max_length, args.device
        )
        print(f"  extraction done in {time.time()-t0:.1f}s")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        print(f"\n[4/4] Fitting linear probes on model representations...")
        for layer in layers:
            feats = features_all[layer]
            X_m_tr = feats[:n_tr]
            X_m_te = feats[n_tr:]
            r = fit_probe(X_m_tr, y_tr, X_m_te, y_te, c_grid, seed=args.seed, scale=True)
            r["feature"] = f"model_layer_{layer:02d}"
            r["n_train"] = len(seqs_tr)
            r["n_test"] = len(seqs_te)
            r["incremental_auroc_vs_best_shortcut"] = r["test_auroc"] - best_shortcut_auroc
            r["incremental_mcc_vs_best_shortcut"] = r["test_mcc"] - best_shortcut_mcc
            results.append(r)
            print_row(f"model_layer_{layer:02d}", r, len(seqs_tr), len(seqs_te))
    else:
        print("\n[3-4/4] Skipping model probe (--skip-model).")

    # ------------------------------------------------------------------
    # Summary and write output
    # ------------------------------------------------------------------
    model_results = [r for r in results if r["feature"].startswith("model_layer")]
    shortcut_results = [r for r in results if not r["feature"].startswith("model_layer")]

    best_shortcut = max(shortcut_results, key=lambda r: r.get("test_sep", 0))
    summary = {
        "task": args.task_name,
        "manifest": args.manifest,
        "n_train": len(seqs_tr),
        "n_test": len(seqs_te),
        "train_label_counts": {str(k): int(v) for k, v in Counter(y_tr.tolist()).items()},
        "test_label_counts": {str(k): int(v) for k, v in Counter(y_te.tolist()).items()},
        "shortcut_baselines": {
            "best_feature": best_shortcut["feature"],
            "best_test_auroc": best_shortcut["test_auroc"],
            "best_test_sep": best_shortcut["test_sep"],
            "best_test_mcc": best_shortcut["test_mcc"],
            "all": shortcut_results,
        },
        "confound_decision": (
            "continue_with_strong_identity_confound_risk"
            if best_shortcut_sep >= 0.90
            else (
                "continue_with_identity_confound_risk"
                if best_shortcut_sep >= 0.80
                else "continue"
            )
        ),
        "formal_success_allowed": best_shortcut_sep < 0.80,
    }

    if model_results:
        best_model = max(model_results, key=lambda r: r.get("test_sep", 0))
        worst_model = min(model_results, key=lambda r: r.get("test_auroc", 1.0))
        summary["model_probes"] = {
            "layers_evaluated": layers,
            "best_layer": best_model["feature"],
            "best_test_auroc": best_model["test_auroc"],
            "best_test_sep": best_model["test_sep"],
            "best_test_mcc": best_model["test_mcc"],
            "min_test_auroc": worst_model["test_auroc"],
            "hidden_mean_auroc": float(np.mean([r["test_auroc"] for r in model_results])),
            "hidden_mean_sep": float(np.mean([r["test_sep"] for r in model_results])),
            "hidden_mean_mcc": float(np.mean([r["test_mcc"] for r in model_results])),
            "incremental_auroc_mean": float(np.mean([r.get("incremental_auroc_vs_best_shortcut", float("nan"))
                                                      for r in model_results])),
            "incremental_mcc_mean": float(np.mean([r.get("incremental_mcc_vs_best_shortcut", float("nan"))
                                                    for r in model_results])),
            "all": model_results,
        }
        print(f"\n{'='*70}")
        print(f"SUMMARY: {args.task_name}")
        print(f"{'='*70}")
        print(f"  Best shortcut AUROC : {best_shortcut['test_auroc']:.4f}  "
              f"sep={best_shortcut['test_sep']:.4f}  MCC={best_shortcut['test_mcc']:.4f}  ({best_shortcut['feature']})")
        print(f"  Best model AUROC    : {best_model['test_auroc']:.4f}  "
              f"sep={best_model['test_sep']:.4f}  MCC={best_model['test_mcc']:.4f}  ({best_model['feature']})")
        print(f"  Incremental AUROC   : {summary['model_probes']['incremental_auroc_mean']:+.4f} (model − best shortcut, mean across layers)")
        print(f"  Incremental MCC     : {summary['model_probes']['incremental_mcc_mean']:+.4f}")
        print(f"  Confound decision   : {summary['confound_decision']}")
        print(f"  Formal success OK   : {summary['formal_success_allowed']}")

    # Write results
    out_json = out_dir / "shortcut_vs_probe_summary.json"
    out_csv = out_dir / "shortcut_vs_probe_all_results.csv"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    fieldnames = sorted({k for r in results for k in r})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  wrote {out_json}")
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    main()
