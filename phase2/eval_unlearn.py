"""
Evaluate internal diagnostics for an unlearned checkpoint by:
  1. Loading base Evo + applying weight deltas from the checkpoint
  2. Extracting mean-pooled activations per layer on the manifest (val+test)
  3. Applying Phase 1 host-tropism probes and reporting AUROC delta vs Phase 1 baseline
  4. Reporting forget/retain perplexity diagnostics on the val split

Primary selective-unlearning evaluation is implemented in phase2/eval_benchmarks.py
for external HVUE/GUE benchmarks.

Outputs:
  data/phase2/checkpoints/<run>/eval_auroc.csv      diagnostic per-layer AUROC after unlearning
  data/phase2/checkpoints/<run>/eval_ppl.json       diagnostic forget vs retain perplexity
"""
import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint, read_manifest
from evo.tokenizer import CharLevelTokenizer
from phase2.utils import get_localized_layers, language_model_loss, tokenize_batch


def apply_checkpoint(model, ckpt_path: str) -> None:
    """Load a checkpoint that contains a *subset* of state_dict keys and apply them."""
    delta = load_file(ckpt_path)
    sd = model.state_dict()
    missing = []
    for key, val in delta.items():
        if key not in sd:
            missing.append(key)
            continue
        sd[key].copy_(val.to(sd[key].dtype).to(sd[key].device))
    if missing:
        print(f"[eval] {len(missing)} keys in ckpt not in model (skipped)")
    print(f"[eval] applied {len(delta) - len(missing)} weight tensors from {ckpt_path}")


def load_probe(probe_dir: str, layer_idx: int):
    data = np.load(os.path.join(probe_dir, f"layer_{layer_idx}.npz"))
    return {
        "coef": data["coef"].astype(np.float32),
        "intercept": data["intercept"].astype(np.float32),
        "mean": data["scaler_mean"].astype(np.float32),
        "scale": data["scaler_scale"].astype(np.float32),
    }


def probe_probs(features: np.ndarray, probe: dict) -> np.ndarray:
    x = (features - probe["mean"]) / np.clip(probe["scale"], 1e-12, None)
    logits = x @ probe["coef"].T + probe["intercept"]
    return 1.0 / (1.0 + np.exp(-logits)).reshape(-1)


def extract_features_for_layers(model, sequences: List[str], tokenizer: CharLevelTokenizer,
                                layers: List[int], batch_size: int, max_length: int,
                                device: str) -> Dict[int, np.ndarray]:
    """Extract mean-pooled activations using next_norm representation (matching Phase 1)."""
    num_layers = len(model.blocks)
    feature_buffers: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    state = {"mask": None}

    handles = []
    layers_set = set(layers)

    def make_hook(layer_idx):
        def hook(_m, _inp, output):
            if layer_idx not in layers_set:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            # next_norm representation
            if layer_idx + 1 < num_layers:
                hidden = model.blocks[layer_idx + 1].pre_norm(hidden)
            else:
                hidden = model.norm(hidden)
            mask = state["mask"]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            feature_buffers[layer_idx].append(pooled.detach().float().cpu().numpy())
        return hook

    for layer_idx in layers:
        handles.append(model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="extract"):
            batch = sequences[start : start + batch_size]
            ids, mask = tokenize_batch(batch, tokenizer, max_length, device)
            state["mask"] = mask
            _ = model(ids, padding_mask=mask)

    for h in handles:
        h.remove()
    return {l: np.concatenate(feature_buffers[l], axis=0) for l in layers}


def measure_perplexity(model, records, tokenizer, batch_size, max_length, device) -> float:
    losses = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            ids, mask = tokenize_batch([r.sequence for r in batch], tokenizer, max_length, device)
            logits, _ = model(ids, padding_mask=mask)
            losses.append(language_model_loss(logits, ids, mask).item())
    mean_loss = float(np.mean(losses))
    return float(np.exp(mean_loss)), mean_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to weights.safetensors")
    parser.add_argument("--manifest", default="data/family_targets/coronaviridae/manifest.csv")
    parser.add_argument("--probe-dir", default="data/family_targets/coronaviridae/probes")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layers", default="0-10",
                        help="Comma list/ranges. Default 0-10 (only stable layers).")
    parser.add_argument("--max-eval", type=int, default=400,
                        help="Cap eval samples per (split,label) to keep runs short.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--localized-layers-path",
        default="data/family_targets/coronaviridae/localized_layers.json",
    )
    args = parser.parse_args()

    # Parse layers
    layers: List[int] = []
    for part in args.layers.split(","):
        part = part.strip()
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            layers.extend(range(a, b + 1))
        elif part:
            layers.append(int(part))
    layers = sorted(set(layers))

    # Load model and apply checkpoint
    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    apply_checkpoint(model, args.ckpt)
    model.eval()
    tokenizer = CharLevelTokenizer(512)

    # Load records (val+test only — train was used for unlearning, would be biased)
    records = read_manifest(args.manifest)
    eval_records = [r for r in records if r.split in ("val", "test")]

    # Cap per (split, label)
    rng = np.random.default_rng(args.seed)
    buckets = {}
    for r in eval_records:
        buckets.setdefault((r.split, r.label), []).append(r)
    capped = []
    for key, recs in buckets.items():
        if len(recs) > args.max_eval:
            idx = rng.choice(len(recs), size=args.max_eval, replace=False)
            recs = [recs[i] for i in idx]
        capped.extend(recs)
    eval_records = capped
    print(f"[eval] {len(eval_records)} eval records "
          f"({sum(r.label==1 for r in eval_records)} forget, "
          f"{sum(r.label==0 for r in eval_records)} retain)")

    # Extract features and score probes
    sequences = [r.sequence for r in eval_records]
    splits = np.array([r.split for r in eval_records])
    labels = np.array([r.label for r in eval_records])
    features = extract_features_for_layers(
        model, sequences, tokenizer, layers, args.batch_size, args.max_length, args.device
    )

    rows = []
    for layer_idx in layers:
        probe = load_probe(args.probe_dir, layer_idx)
        feats = features[layer_idx]
        x = (feats - probe["mean"]) / np.clip(probe["scale"], 1e-12, None)
        logits = x @ probe["coef"].T + probe["intercept"]
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = probs.reshape(-1)
        preds = (probs >= 0.5).astype(np.int64)
        row = {"layer": layer_idx}
        for split in ("val", "test"):
            mask = splits == split
            if mask.sum() == 0:
                continue
            row[f"{split}_acc"] = float(accuracy_score(labels[mask], preds[mask]))
            row[f"{split}_mcc"] = float(matthews_corrcoef(labels[mask], preds[mask]))
            row[f"{split}_auroc"] = float(roc_auc_score(labels[mask], probs[mask]))
        rows.append(row)
        print(f"  layer {layer_idx:>2}: "
              f"val_auroc={row.get('val_auroc', 0):.4f}  test_auroc={row.get('test_auroc', 0):.4f}")

    localized_layers = [layer for layer in get_localized_layers(args.localized_layers_path) if layer in layers]
    localized_summary = {}
    if localized_layers:
        val_scores = [row["val_auroc"] for row in rows if row["layer"] in localized_layers and "val_auroc" in row]
        test_scores = [row["test_auroc"] for row in rows if row["layer"] in localized_layers and "test_auroc" in row]
        localized_summary = {
            "localized_layers": localized_layers,
            "localized_val_mean_auroc": float(np.mean(val_scores)) if val_scores else None,
            "localized_test_mean_auroc": float(np.mean(test_scores)) if test_scores else None,
        }
        print(
            "[eval] localized mean AUROC "
            f"val={localized_summary['localized_val_mean_auroc']:.4f} "
            f"test={localized_summary['localized_test_mean_auroc']:.4f}"
        )

    out_dir = os.path.dirname(args.ckpt)
    auroc_path = os.path.join(out_dir, "eval_auroc.csv")
    with open(auroc_path, "w", newline="") as f:
        fieldnames = [
            "layer", "val_acc", "val_mcc", "val_auroc",
            "test_acc", "test_mcc", "test_auroc",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[eval] wrote AUROC to {auroc_path}")

    # Perplexity on val split
    forget_val = [r for r in eval_records if r.label == 1 and r.split == "val"]
    retain_val = [r for r in eval_records if r.label == 0 and r.split == "val"]
    fppl, floss = measure_perplexity(model, forget_val, tokenizer, args.batch_size, args.max_length, args.device)
    rppl, rloss = measure_perplexity(model, retain_val, tokenizer, args.batch_size, args.max_length, args.device)
    ppl_path = os.path.join(out_dir, "eval_ppl.json")
    with open(ppl_path, "w") as f:
        payload = {
            "forget_val_perplexity": fppl, "forget_val_loss": floss, "n_forget": len(forget_val),
            "retain_val_perplexity": rppl, "retain_val_loss": rloss, "n_retain": len(retain_val),
        }
        payload.update(localized_summary)
        json.dump(payload, f, indent=2)
    print(f"[eval] forget_ppl={fppl:.3f}  retain_ppl={rppl:.3f}")
    print(f"[eval] wrote perplexity to {ppl_path}")

    # Fixed held-out representation comparison. Training-log representation
    # metrics are batch-dependent and must not be used for checkpoint selection.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base_model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    base_model.eval()
    base_features = extract_features_for_layers(
        base_model, sequences, tokenizer, layers, args.batch_size, args.max_length, args.device
    )
    rep_rows = []
    for layer_idx in layers:
        original = base_features[layer_idx].astype(np.float32)
        modified = features[layer_idx].astype(np.float32)
        for split in ("val", "test"):
            for label, subset in (("forget", 1), ("retain", 0)):
                mask = (splits == split) & (labels == subset)
                if not mask.any():
                    continue
                a, b = original[mask], modified[mask]
                mse_per_example = np.mean((a - b) ** 2, axis=1)
                denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                cosine = np.sum(a * b, axis=1) / np.clip(denom, 1e-12, None)
                rep_rows.append({
                    "layer": layer_idx,
                    "split": split,
                    "subset": label,
                    "n": int(mask.sum()),
                    "representation_mse": float(np.mean(mse_per_example)),
                    "original_modified_cosine": float(np.mean(cosine)),
                })
    rep_path = os.path.join(out_dir, "eval_representation.csv")
    rep_fields = [
        "layer", "split", "subset", "n",
        "representation_mse", "original_modified_cosine",
    ]
    with open(rep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rep_fields)
        writer.writeheader()
        writer.writerows(rep_rows)
    print(f"[eval] wrote held-out representation metrics to {rep_path}")


if __name__ == "__main__":
    main()
