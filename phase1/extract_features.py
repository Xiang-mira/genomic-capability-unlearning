import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import FeatureWriter, read_manifest, load_local_checkpoint, pad_batch
from evo.tokenizer import CharLevelTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layer-wise mean-pooled features from Evo.")
<<<<<<< HEAD
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/host_tropism/features")
=======
    parser.add_argument("--manifest", default="data/phase1/manifest.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase1/features")
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
<<<<<<< HEAD
    parser.add_argument(
        "--representation",
        choices=["raw", "next_norm"],
        default="raw",
        help=(
            "Feature representation to pool. raw pools block outputs. next_norm "
            "applies the next block's pre_norm, or final model.norm for the last layer."
        ),
    )
=======
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    sequences = [r.sequence for r in records]
    labels = np.array([r.label for r in records], dtype=np.int64)
    splits = np.array([r.split for r in records])
    record_ids = np.array([r.record_id for r in records])

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    np.save(os.path.join(args.out_dir, "splits.npy"), splits)
    np.save(os.path.join(args.out_dir, "ids.npy"), record_ids)

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.eval()

    tokenizer = CharLevelTokenizer(512)

    num_layers = len(model.blocks)
    writer = FeatureWriter(args.out_dir, num_layers=num_layers, chunk_size=args.chunk_size)

<<<<<<< HEAD
    block_to_indices = {}
    for layer_idx, block in enumerate(model.blocks):
        block_to_indices.setdefault(block, []).append(layer_idx)

    state: Dict[str, torch.Tensor] = {"mask": None, "call_counts": None}

    def make_hook(block):
        indices = block_to_indices[block]

        def hook(_module, _inputs, output):
            call_counts = state["call_counts"]
            if call_counts is None:
                return
            call_idx = call_counts[block]
            if call_idx >= len(indices):
                raise RuntimeError(
                    "Block hook called more times than expected for a single forward."
                )
            layer_idx = indices[call_idx]
            call_counts[block] = call_idx + 1
            hidden = output[0] if isinstance(output, tuple) else output
            if args.representation == "next_norm":
                if layer_idx + 1 < num_layers:
                    hidden = model.blocks[layer_idx + 1].pre_norm(hidden)
                else:
                    hidden = model.norm(hidden)
=======
    state: Dict[str, torch.Tensor] = {"mask": None}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
            mask = state["mask"]
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            writer.add(layer_idx, pooled.detach().float().cpu())
        return hook

<<<<<<< HEAD
    hooks = [block.register_forward_hook(make_hook(block)) for block in block_to_indices]
=======
    hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(model.blocks)]
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), args.batch_size), desc="Extracting"):
            batch = sequences[start : start + args.batch_size]
            batch = [seq[: args.max_length] for seq in batch]
            token_ids = tokenizer.tokenize_batch(batch)
            input_ids, mask = pad_batch(token_ids, tokenizer.pad_id)
            input_ids = input_ids.to(args.device)
            state["mask"] = mask.to(args.device)
<<<<<<< HEAD
            state["call_counts"] = {block: 0 for block in block_to_indices}
            _ = model(input_ids, padding_mask=state["mask"])
            total_calls = sum(state["call_counts"].values())
            state["call_counts"] = None
            if total_calls != num_layers:
                raise RuntimeError(
                    f"Expected {num_layers} block calls, got {total_calls}."
                )
=======
            _ = model(input_ids, padding_mask=state["mask"])
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87

    writer.flush_all()
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
