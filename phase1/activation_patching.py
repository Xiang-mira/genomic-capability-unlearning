import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import ManifestRecord, load_local_checkpoint, read_manifest


@dataclass
class PairBatch:
    source_ids: torch.Tensor
    source_mask: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor


def select_records(
    records: List[ManifestRecord],
    split: str,
    label: int,
    n: int,
    seed: int,
) -> List[ManifestRecord]:
    candidates = [record for record in records if record.split == split and record.label == label]
    if len(candidates) < n:
        raise ValueError(
            f"Need {n} records for split={split} label={label}, found {len(candidates)}."
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def make_pairs(
    records: List[ManifestRecord],
    split: str,
    source_label: int,
    target_label: int,
    n_pairs: int,
    seed: int,
) -> List[Tuple[ManifestRecord, ManifestRecord]]:
    sources = select_records(records, split, source_label, n_pairs, seed)
    targets = select_records(records, split, target_label, n_pairs, seed + 1)
    return list(zip(sources, targets))


def pad_to_common_length(
    token_ids: List[List[int]],
    pad_id: int,
    length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = np.full((len(token_ids), length), pad_id, dtype=np.int64)
    mask = np.zeros((len(token_ids), length), dtype=np.int64)
    for i, tokens in enumerate(token_ids):
        clipped = tokens[:length]
        batch[i, : len(clipped)] = clipped
        mask[i, : len(clipped)] = 1
    return torch.from_numpy(batch), torch.from_numpy(mask)


def iter_pair_batches(
    pairs: List[Tuple[ManifestRecord, ManifestRecord]],
    tokenizer: CharLevelTokenizer,
    batch_size: int,
    max_length: int,
    device: str,
) -> Iterable[PairBatch]:
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        source_tokens = [
            tokenizer.tokenize(record.sequence[:max_length]) for record, _target in batch
        ]
        target_tokens = [
            tokenizer.tokenize(record.sequence[:max_length]) for _source, record in batch
        ]
        common_length = min(
            max_length,
            max(max(len(tokens) for tokens in source_tokens), max(len(tokens) for tokens in target_tokens)),
        )
        source_ids, source_mask = pad_to_common_length(source_tokens, tokenizer.pad_id, common_length)
        target_ids, target_mask = pad_to_common_length(target_tokens, tokenizer.pad_id, common_length)
        yield PairBatch(
            source_ids=source_ids.to(device),
            source_mask=source_mask.to(device),
            target_ids=target_ids.to(device),
            target_mask=target_mask.to(device),
        )


def language_model_loss(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous().bool()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    denom = shift_mask.sum(dim=1).clamp(min=1)
    return (token_losses * shift_mask).sum(dim=1) / denom


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (hidden.float() * mask.unsqueeze(-1).float()).sum(dim=1) / denom


def load_probe(probe_dir: str, layer_idx: int, device: str) -> Dict[str, torch.Tensor]:
    path = os.path.join(probe_dir, f"layer_{layer_idx}.npz")
    data = np.load(path)
    return {
        "coef": torch.from_numpy(data["coef"].astype(np.float32)).to(device),
        "intercept": torch.from_numpy(data["intercept"].astype(np.float32)).to(device),
        "mean": torch.from_numpy(data["scaler_mean"].astype(np.float32)).to(device),
        "scale": torch.from_numpy(data["scaler_scale"].astype(np.float32)).to(device),
    }


def probe_probability(pooled: torch.Tensor, probe: Dict[str, torch.Tensor]) -> torch.Tensor:
    scaled = (pooled - probe["mean"]) / probe["scale"].clamp(min=1e-12)
    logits = scaled @ probe["coef"].t() + probe["intercept"]
    return torch.sigmoid(logits.squeeze(-1))


def pooled_distances(source_pooled: torch.Tensor, target_pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    l2 = torch.linalg.vector_norm(source_pooled - target_pooled, dim=1)
    cosine = F.cosine_similarity(source_pooled.float(), target_pooled.float(), dim=1)
    return l2, 1.0 - cosine


class ActivationPatcher:
    def __init__(self, model, num_layers: int) -> None:
        self.model = model
        self.num_layers = num_layers
        self.cache: Dict[int, torch.Tensor] = {}
        self.captured: Dict[int, torch.Tensor] = {}
        self.patch_layer: int | None = None
        self.mode = "capture"
        self.source_mask: torch.Tensor | None = None
        self.target_mask: torch.Tensor | None = None
        self.handles = []

        block_to_indices = {}
        for layer_idx, block in enumerate(model.blocks):
            block_to_indices.setdefault(block, []).append(layer_idx)
        self.block_to_indices = block_to_indices
        self.call_counts: Dict[torch.nn.Module, int] | None = None

    def __enter__(self):
        for block in self.block_to_indices:
            self.handles.append(block.register_forward_hook(self._make_hook(block)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()

    def _make_hook(self, block):
        def hook(_module, _inputs, output):
            if self.call_counts is None:
                return output
            call_idx = self.call_counts[block]
            indices = self.block_to_indices[block]
            if call_idx >= len(indices):
                raise RuntimeError("Block hook called more times than expected.")
            layer_idx = indices[call_idx]
            self.call_counts[block] = call_idx + 1

            hidden = output[0] if isinstance(output, tuple) else output
            if self.mode == "source":
                self.cache[layer_idx] = hidden.detach()
                return output
            if self.mode == "clean":
                self.captured[layer_idx] = hidden.detach()
                return output
            if self.mode == "patch" and layer_idx == self.patch_layer:
                source = self.cache[layer_idx].to(hidden.device)
                if source.shape != hidden.shape:
                    raise RuntimeError(
                        f"Shape mismatch at layer {layer_idx}: source={source.shape} target={hidden.shape}"
                    )
                if self.source_mask is None or self.target_mask is None:
                    patch_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                else:
                    patch_mask = (self.source_mask.bool() & self.target_mask.bool()).to(hidden.device)
                patched = torch.where(patch_mask.unsqueeze(-1), source, hidden)
                self.captured[layer_idx] = patched.detach()
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched
            return output

        return hook

    def begin_forward(self, mode: str, patch_layer: int | None = None) -> None:
        self.mode = mode
        self.patch_layer = patch_layer
        self.call_counts = {block: 0 for block in self.block_to_indices}
        if mode in {"source", "clean"}:
            self.captured = {}
        if mode == "source":
            self.cache = {}

    def end_forward(self) -> None:
        if self.call_counts is None:
            return
        total_calls = sum(self.call_counts.values())
        self.call_counts = None
        if total_calls != self.num_layers:
            raise RuntimeError(f"Expected {self.num_layers} block calls, got {total_calls}.")


def evaluate_direction(
    model,
    patcher: ActivationPatcher,
    pairs: List[Tuple[ManifestRecord, ManifestRecord]],
    tokenizer: CharLevelTokenizer,
    probes: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    args,
    direction: str,
) -> List[Dict[str, float | int | str]]:
    rows = []
    totals = {
        layer: {
            "n": 0,
            "clean_loss": 0.0,
            "patched_loss": 0.0,
            "clean_prob": 0.0,
            "patched_prob": 0.0,
        }
        for layer in layers
    }

    with torch.no_grad():
        iterator = iter_pair_batches(pairs, tokenizer, args.batch_size, args.max_length, args.device)
        for batch in tqdm(iterator, total=(len(pairs) + args.batch_size - 1) // args.batch_size, desc=direction):
            patcher.source_mask = batch.source_mask
            patcher.target_mask = batch.target_mask

            patcher.begin_forward("source")
            _source_logits, _ = model(batch.source_ids, padding_mask=batch.source_mask)
            patcher.end_forward()
            source_hidden = {layer_idx: patcher.cache[layer_idx] for layer_idx in layers}

            patcher.begin_forward("clean")
            clean_logits, _ = model(batch.target_ids, padding_mask=batch.target_mask)
            patcher.end_forward()
            clean_loss = language_model_loss(clean_logits, batch.target_ids, batch.target_mask)

            clean_probs = {}
            activation_l2 = {}
            activation_cosine_distance = {}
            for layer_idx in layers:
                clean_hidden = patcher.captured[layer_idx]
                clean_pooled = masked_mean(clean_hidden, batch.target_mask)
                clean_probs[layer_idx] = probe_probability(clean_pooled, probes[layer_idx])
                source_pooled = masked_mean(source_hidden[layer_idx], batch.source_mask)
                l2, cosine_distance = pooled_distances(source_pooled, clean_pooled)
                activation_l2[layer_idx] = l2
                activation_cosine_distance[layer_idx] = cosine_distance

            for layer_idx in layers:
                patcher.begin_forward("patch", patch_layer=layer_idx)
                patched_logits, _ = model(batch.target_ids, padding_mask=batch.target_mask)
                patcher.end_forward()
                patched_loss = language_model_loss(patched_logits, batch.target_ids, batch.target_mask)
                patched_hidden = patcher.captured[layer_idx]
                patched_pooled = masked_mean(patched_hidden, batch.target_mask)
                patched_prob = probe_probability(patched_pooled, probes[layer_idx])

                n = batch.target_ids.shape[0]
                totals[layer_idx]["n"] += n
                totals[layer_idx]["clean_loss"] += clean_loss.sum().item()
                totals[layer_idx]["patched_loss"] += patched_loss.sum().item()
                totals[layer_idx]["clean_prob"] += clean_probs[layer_idx].sum().item()
                totals[layer_idx]["patched_prob"] += patched_prob.sum().item()
                totals[layer_idx].setdefault("activation_l2", 0.0)
                totals[layer_idx].setdefault("activation_cosine_distance", 0.0)
                totals[layer_idx]["activation_l2"] += activation_l2[layer_idx].sum().item()
                totals[layer_idx]["activation_cosine_distance"] += (
                    activation_cosine_distance[layer_idx].sum().item()
                )

    for layer_idx in layers:
        total = totals[layer_idx]
        n = total["n"]
        clean_loss = total["clean_loss"] / n
        patched_loss = total["patched_loss"] / n
        clean_prob = total["clean_prob"] / n
        patched_prob = total["patched_prob"] / n
        clean_perplexity = float(np.exp(clean_loss))
        patched_perplexity = float(np.exp(patched_loss))
        rows.append(
            {
                "direction": direction,
                "source_label": pairs[0][0].label,
                "target_label": pairs[0][1].label,
                "layer": layer_idx,
                "n_pairs": n,
                "clean_loss": clean_loss,
                "patched_loss": patched_loss,
                "delta_loss": patched_loss - clean_loss,
                "clean_perplexity": clean_perplexity,
                "patched_perplexity": patched_perplexity,
                "delta_perplexity": patched_perplexity - clean_perplexity,
                "clean_human_prob": clean_prob,
                "patched_human_prob": patched_prob,
                "delta_human_prob": patched_prob - clean_prob,
                "abs_delta_human_prob": abs(patched_prob - clean_prob),
                "activation_l2": total["activation_l2"] / n,
                "activation_cosine_distance": total["activation_cosine_distance"] / n,
            }
        )
    return rows


def parse_layers(value: str, num_layers: int) -> List[int]:
    if value == "all":
        return list(range(num_layers))
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} is outside valid range [0, {num_layers - 1}].")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise activation patching for human vs non-human viral host tropism."
    )
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--probe-dir", default="data/host_tropism/probes")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/host_tropism/activation_patching")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="all", help="Comma list/ranges, e.g. all or 0-10,14,21.")
    parser.add_argument(
        "--directions",
        choices=["both", "nonhuman_to_human", "human_to_nonhuman"],
        default="both",
    )
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.eval()
    tokenizer = CharLevelTokenizer(512)
    num_layers = len(model.blocks)
    layers = parse_layers(args.layers, num_layers)
    probes = {layer: load_probe(args.probe_dir, layer, args.device) for layer in layers}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "patching_by_layer.csv")
    summary_path = os.path.join(args.out_dir, "patching_layer_summary.csv")
    rows = []
    with ActivationPatcher(model, num_layers) as patcher:
        if args.directions in {"both", "nonhuman_to_human"}:
            pairs = make_pairs(records, args.split, source_label=0, target_label=1, n_pairs=args.n_pairs, seed=args.seed)
            rows.extend(
                evaluate_direction(
                    model, patcher, pairs, tokenizer, probes, layers, args, "nonhuman_to_human"
                )
            )
        if args.directions in {"both", "human_to_nonhuman"}:
            pairs = make_pairs(records, args.split, source_label=1, target_label=0, n_pairs=args.n_pairs, seed=args.seed + 1000)
            rows.extend(
                evaluate_direction(
                    model, patcher, pairs, tokenizer, probes, layers, args, "human_to_nonhuman"
                )
            )

    fieldnames = [
        "direction",
        "source_label",
        "target_label",
        "layer",
        "n_pairs",
        "clean_loss",
        "patched_loss",
        "delta_loss",
        "clean_perplexity",
        "patched_perplexity",
        "delta_perplexity",
        "clean_human_prob",
        "patched_human_prob",
        "delta_human_prob",
        "abs_delta_human_prob",
        "activation_l2",
        "activation_cosine_distance",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote activation patching metrics to {out_path}")

    summary_rows = []
    for layer in layers:
        layer_rows = [row for row in rows if row["layer"] == layer]
        summary_rows.append(
            {
                "layer": layer,
                "mean_abs_delta_human_prob": float(
                    np.mean([row["abs_delta_human_prob"] for row in layer_rows])
                ),
                "mean_abs_delta_loss": float(np.mean([abs(row["delta_loss"]) for row in layer_rows])),
                "mean_abs_delta_perplexity": float(
                    np.mean([abs(row["delta_perplexity"]) for row in layer_rows])
                ),
                "mean_activation_l2": float(np.mean([row["activation_l2"] for row in layer_rows])),
                "mean_activation_cosine_distance": float(
                    np.mean([row["activation_cosine_distance"] for row in layer_rows])
                ),
            }
        )
    summary_rows.sort(
        key=lambda row: (row["mean_abs_delta_human_prob"], row["mean_abs_delta_loss"]),
        reverse=True,
    )
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "mean_abs_delta_human_prob",
                "mean_abs_delta_loss",
                "mean_abs_delta_perplexity",
                "mean_activation_l2",
                "mean_activation_cosine_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote layer ranking summary to {summary_path}")
    print("Top layers by patch effect:")
    for row in summary_rows[: min(10, len(summary_rows))]:
        print(
            f"  layer {row['layer']}: "
            f"abs_delta_prob={row['mean_abs_delta_human_prob']:.6f} "
            f"abs_delta_loss={row['mean_abs_delta_loss']:.6f} "
            f"abs_delta_ppl={row['mean_abs_delta_perplexity']:.6f}"
        )


if __name__ == "__main__":
    main()
