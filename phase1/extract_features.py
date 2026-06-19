import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import FeatureWriter, load_local_checkpoint, pad_batch, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layer-wise mean-pooled features from Evo.")
    parser.add_argument("--manifest", default="data/family_targets/coronaviridae/manifest.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/family_targets/coronaviridae/features")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--representation",
        choices=["raw", "next_norm"],
        default="next_norm",
        help=(
            "Feature representation to pool. raw pools block outputs. next_norm applies "
            "the next block's pre_norm, or final model.norm for the last layer."
        ),
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma list/ranges of layers to extract, e.g. all or 0-10,14.",
    )
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
    if args.layers == "all":
        layers = list(range(num_layers))
    else:
        layers = []
        for part in args.layers.split(","):
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

    writer = FeatureWriter(args.out_dir, num_layers=num_layers, chunk_size=args.chunk_size)

    block_to_indices = {}
    for layer_idx, block in enumerate(model.blocks):
        block_to_indices.setdefault(block, []).append(layer_idx)

    state: Dict[str, torch.Tensor | Dict[torch.nn.Module, int] | None] = {
        "mask": None,
        "call_counts": None,
    }
    layer_set = set(layers)

    def make_hook(block):
        indices = block_to_indices[block]

        def hook(_module, _inputs, output):
            call_counts = state["call_counts"]
            if call_counts is None:
                return
            layer_idx = indices[call_counts[block]]
            call_counts[block] += 1
            if layer_idx not in layer_set:
                return

            hidden = output[0] if isinstance(output, tuple) else output
            if args.representation == "next_norm":
                if layer_idx + 1 < num_layers:
                    hidden = model.blocks[layer_idx + 1].pre_norm(hidden)
                else:
                    hidden = model.norm(hidden)

            mask = state["mask"]
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            writer.add(layer_idx, pooled.detach().float().cpu())

        return hook

    hooks = [block.register_forward_hook(make_hook(block)) for block in block_to_indices]

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), args.batch_size), desc="Extracting"):
            batch = [seq[: args.max_length] for seq in sequences[start : start + args.batch_size]]
            token_ids = tokenizer.tokenize_batch(batch)
            input_ids, mask = pad_batch(token_ids, tokenizer.pad_id)
            input_ids = input_ids.to(args.device)
            state["mask"] = mask.to(args.device)
            state["call_counts"] = {block: 0 for block in block_to_indices}
            _ = model(input_ids, padding_mask=state["mask"])
            total_calls = sum(state["call_counts"].values())
            state["call_counts"] = None
            if total_calls != num_layers:
                raise RuntimeError(f"Expected {num_layers} block calls, got {total_calls}.")

    writer.flush_all()
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
