"""Shared utilities for Phase 3 recovery attacks."""
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint, read_manifest
from evo.tokenizer import CharLevelTokenizer
from phase2.utils import language_model_loss, tokenize_batch


def apply_checkpoint(model, ckpt_path: str) -> None:
    delta = load_file(ckpt_path)
    sd = model.state_dict()
    for key, val in delta.items():
        if key in sd:
            sd[key].copy_(val.to(sd[key].dtype).to(sd[key].device))
    print(f"[phase3] applied {len(delta)} tensors from {ckpt_path}")


def load_probe(probe_dir: str, layer_idx: int) -> Dict[str, np.ndarray]:
    data = np.load(os.path.join(probe_dir, f"layer_{layer_idx}.npz"))
    return {k: data[k].astype(np.float32) for k in data.files}


def extract_features(model, sequences: List[str], tokenizer, layers: List[int],
                     batch_size: int, max_length: int, device: str) -> Dict[int, np.ndarray]:
    num_layers = len(model.blocks)
    buffers: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    state = {"mask": None}

    handles = []
    for layer_idx in layers:
        def make_hook(li):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if li + 1 < num_layers:
                    h = model.blocks[li + 1].pre_norm(h)
                else:
                    h = model.norm(h)
                mask = state["mask"]
                denom = mask.sum(1, keepdim=True).clamp(min=1)
                pooled = (h * mask.unsqueeze(-1)).sum(1) / denom
                buffers[li].append(pooled.detach().float().cpu().numpy())
            return hook
        handles.append(model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start: start + batch_size]
            ids, mask = tokenize_batch(batch, tokenizer, max_length, device)
            state["mask"] = mask
            _ = model(ids, padding_mask=mask)

    for h in handles:
        h.remove()
    return {l: np.concatenate(buffers[l], axis=0) for l in layers}


def probe_auroc(features: np.ndarray, labels: np.ndarray, probe: Dict) -> float:
    x = (features - probe["scaler_mean"]) / np.clip(probe["scaler_scale"], 1e-12, None)
    logits = x @ probe["coef"].T + probe["intercept"]
    probs = 1.0 / (1.0 + np.exp(-logits.clip(-500, 500))).reshape(-1)
    return float(roc_auc_score(labels, probs))


def eval_auroc_all_layers(model, records, tokenizer, probe_dir: str,
                          layers: List[int], batch_size: int, max_length: int,
                          device: str) -> Dict[int, float]:
    sequences = [r.sequence for r in records]
    labels = np.array([r.label for r in records])
    features = extract_features(model, sequences, tokenizer, layers,
                                batch_size, max_length, device)
    result = {}
    for l in layers:
        probe = load_probe(probe_dir, l)
        result[l] = probe_auroc(features[l], labels, probe)
    return result


def write_results(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
