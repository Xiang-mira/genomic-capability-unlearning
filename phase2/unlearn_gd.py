"""
Gradient Difference unlearning for Evo-1-8k-base on a taxonomy-defined target family.

Per step:
  L_forget = CE on a forget batch
  L_retain = CE on a retain batch
  loss = -alpha_forget * L_forget + alpha_retain * L_retain
  (maximize forget loss, minimize retain loss)

Condition controls which parameter groups receive grad:
  full       : entire model
  localized  : layers loaded from localized_layers.json (causal layers from patching)
  random     : 7 randomly sampled blocks from layers 11..30
"""
import argparse
import json
import os
import random
import sys
import time
from typing import List

import torch
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint, read_manifest
from evo.tokenizer import CharLevelTokenizer
from phase2.utils import (
    PROBE_LAYERS,
    count_trainable,
    freeze_all,
    get_localized_layers,
    iterate_batches,
    language_model_loss,
    select_random_layers,
    set_block_grad,
    split_records,
    tokenize_batch,
)


def parse_save_steps(spec: str) -> set[int]:
    if not spec.strip():
        return set()
    steps = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        steps.add(int(part))
    return steps


def filter_train(records):
    return [r for r in records if r.split == "train"]


def configure_trainable_blocks(model, condition: str, seed: int, localized_layers_path: str) -> List[int]:
    localized_layers = get_localized_layers(localized_layers_path)
    freeze_all(model)
    if condition == "full":
        layers = list(range(len(model.blocks)))
        for p in model.parameters():
            p.requires_grad_(True)
    elif condition == "localized":
        layers = localized_layers
        set_block_grad(model, layers, True)
    elif condition == "probe":
        layers = PROBE_LAYERS
        set_block_grad(model, layers, True)
    elif condition == "random":
        layers = select_random_layers(seed, n=len(localized_layers))
        set_block_grad(model, layers, True)
    else:
        raise ValueError(f"Unknown condition {condition}")
    return layers


def save_block_deltas(model, ref_state, layers: List[int], out_path: str) -> None:
    """Save only the (modified - reference) deltas for trained blocks.
    Keeps checkpoints compact: 7 blocks ~ 200M params * 2 bytes ~ 400 MB.
    """
    delta = {}
    sd = model.state_dict()
    for layer_idx in layers:
        prefix = f"blocks.{layer_idx}."
        for key, val in sd.items():
            if key.startswith(prefix):
                delta[key] = val.detach().to(torch.bfloat16).cpu()
    # also save embedding/unembed if full
    if len(layers) == len(model.blocks):
        # full-model: store the full state dict via save_file
        for key, val in sd.items():
            delta[key] = val.detach().to(torch.bfloat16).cpu()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(delta, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget-csv", default="data/phase2/coronaviridae_splits/forget.csv")
    parser.add_argument("--retain-csv", default="data/phase2/coronaviridae_splits/retain.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase2/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=["full", "localized", "probe", "random"], required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--alpha-forget", type=float, default=1.0)
    parser.add_argument("--alpha-retain", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-steps",
        default="100,200,500,1000",
        help="Comma-separated training steps to save as intermediate checkpoints, e.g. 100,200,500,1000.",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the output directory name (default: gd_<condition>).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--localized-layers-path",
        default="data/family_targets/coronaviridae/localized_layers.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print(f"[GD] condition={args.condition} steps={args.steps} lr={args.lr}")
    save_steps = {step for step in parse_save_steps(args.save_steps) if 1 <= step <= args.steps}
    run_name = args.run_name if args.run_name else f"gd_{args.condition}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if save_steps:
        print(f"[GD] intermediate save steps: {sorted(save_steps)}")

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.train()
    tokenizer = CharLevelTokenizer(512)

    layers = configure_trainable_blocks(model, args.condition, args.seed, args.localized_layers_path)
    print(f"[GD] trainable layers: {layers}")
    print(f"[GD] trainable params: {count_trainable(model):,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    forget = filter_train(read_manifest(args.forget_csv))
    retain = filter_train(read_manifest(args.retain_csv))
    print(f"[GD] forget train={len(forget)}  retain train={len(retain)}")

    log_rows = []
    forget_iter = iter([])
    retain_iter = iter([])

    def next_forget_batch():
        nonlocal forget_iter
        try:
            return next(forget_iter)
        except StopIteration:
            forget_iter = iter(iterate_batches(forget, args.batch_size, shuffle=True, rng=rng))
            return next(forget_iter)

    def next_retain_batch():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(iterate_batches(retain, args.batch_size, shuffle=True, rng=rng))
            return next(retain_iter)

    pbar = tqdm(range(args.steps), desc=f"GD-{args.condition}")
    t0 = time.time()
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        # Forget loss (maximize)
        fbatch = next_forget_batch()
        f_ids, f_mask = tokenize_batch([r.sequence for r in fbatch], tokenizer, args.max_length, args.device)
        f_logits, _ = model(f_ids, padding_mask=f_mask)
        L_forget = language_model_loss(f_logits, f_ids, f_mask)

        # Retain loss (minimize)
        rbatch = next_retain_batch()
        r_ids, r_mask = tokenize_batch([r.sequence for r in rbatch], tokenizer, args.max_length, args.device)
        r_logits, _ = model(r_ids, padding_mask=r_mask)
        L_retain = language_model_loss(r_logits, r_ids, r_mask)

        weighted_forget = -args.alpha_forget * L_forget
        weighted_retain = args.alpha_retain * L_retain
        loss = weighted_forget + weighted_retain
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        current_step = step + 1
        if (step + 1) % args.log_every == 0 or step == 0:
            row = {
                "step": current_step,
                "L_forget": L_forget.item(),
                "L_retain": L_retain.item(),
                "weighted_forget_term": weighted_forget.item(),
                "weighted_retain_term": weighted_retain.item(),
                "loss": loss.item(),
            }
            log_rows.append(row)
            pbar.set_postfix(Lf=f"{row['L_forget']:.3f}", Lr=f"{row['L_retain']:.3f}")
        if current_step in save_steps:
            step_dir = os.path.join(run_dir, f"step_{current_step:06d}")
            save_block_deltas(model, ref_state=None, layers=layers,
                              out_path=os.path.join(step_dir, "weights.safetensors"))
            with open(os.path.join(step_dir, "meta.json"), "w") as f:
                json.dump({
                    "method": "gradient_difference",
                    "condition": args.condition,
                    "layers": layers,
                    "checkpoint_step": current_step,
                    "steps": args.steps, "lr": args.lr,
                    "alpha_forget": args.alpha_forget, "alpha_retain": args.alpha_retain,
                    "batch_size": args.batch_size, "max_length": args.max_length,
                    "forget_csv": args.forget_csv,
                    "retain_csv": args.retain_csv,
                    "seed": args.seed,
                    "parent_run": run_name,
                }, f, indent=2)

    elapsed = time.time() - t0
    print(f"[GD] done in {elapsed:.1f}s")

    # Save deltas
    save_block_deltas(model, ref_state=None, layers=layers,
                      out_path=os.path.join(run_dir, "weights.safetensors"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "method": "gradient_difference",
            "condition": args.condition,
            "layers": layers,
            "steps": args.steps, "lr": args.lr,
            "alpha_forget": args.alpha_forget, "alpha_retain": args.alpha_retain,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "forget_csv": args.forget_csv,
            "retain_csv": args.retain_csv,
            "seed": args.seed, "elapsed_sec": elapsed,
            "save_steps": sorted(save_steps),
            "localized_layers_path": args.localized_layers_path,
        }, f, indent=2)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"[GD] saved to {run_dir}")


if __name__ == "__main__":
    main()
