"""Shared utilities for Phase 2 unlearning."""
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import ManifestRecord, pad_batch, read_manifest
from evo.tokenizer import CharLevelTokenizer


# Layers selected from activation_patching_analysis.md
LOCALIZED_LAYERS = [3, 4, 5, 6, 7, 8, 9]  # 7 layers, causal effect
PROBE_LAYERS = list(range(0, 11))           # layers 0-10, strongest probe AUROC
RANDOM_LAYER_POOL = list(range(11, 31))  # exclude the unstable last block


def select_random_layers(seed: int, n: int) -> List[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(RANDOM_LAYER_POOL, n))


def split_records(records: List[ManifestRecord]) -> Tuple[List[ManifestRecord], List[ManifestRecord]]:
    """label=1 → forget (human-tropic); label=0 → retain (non-human-tropic)."""
    forget = [r for r in records if r.label == 1]
    retain = [r for r in records if r.label == 0]
    return forget, retain


def tokenize_batch(seqs: List[str], tokenizer: CharLevelTokenizer, max_length: int, device: str
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    token_ids = [tokenizer.tokenize(seq[:max_length]) for seq in seqs]
    input_ids, mask = pad_batch(token_ids, tokenizer.pad_id)
    return input_ids.to(device), mask.to(device)


def language_model_loss(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
                        ) -> torch.Tensor:
    """Mean per-sequence next-token CE loss. Returns scalar averaged over batch."""
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous().float()

    flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.view(-1)
    token_losses = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels, reduction="none"
    ).view_as(shift_labels)
    denom = shift_mask.sum(dim=1).clamp(min=1)
    per_seq = (token_losses * shift_mask).sum(dim=1) / denom
    return per_seq.mean()


def iterate_batches(records: List[ManifestRecord], batch_size: int, shuffle: bool, rng: random.Random
                    ) -> Iterable[List[ManifestRecord]]:
    if shuffle:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        records = [records[i] for i in indices]
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if not batch:
            continue
        yield batch


def set_block_grad(model, block_indices: List[int], requires_grad: bool) -> None:
    """Toggle requires_grad for parameters in selected blocks only."""
    for layer_idx in block_indices:
        for p in model.blocks[layer_idx].parameters():
            p.requires_grad_(requires_grad)


def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def get_trainable_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_split_meta(path: str, condition: str, layers: List[int], extra: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"condition,{condition}\n")
        f.write(f"layers,{'|'.join(str(x) for x in layers)}\n")
        for k, v in extra.items():
            f.write(f"{k},{v}\n")
