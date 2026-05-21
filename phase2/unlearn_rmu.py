"""
RMU (Representation Misdirection for Unlearning) for Evo-1-8k-base.

Idea (Li et al., ICML 2024):
  - Forget set: push hidden activations at a target layer toward a random unit
    direction (steering coefficient c). This destroys the structured representation.
  - Retain set: keep hidden activations close to those of a *frozen* reference
    copy of the model.

We use a single target hook layer (default: layer 6, the strongest causal layer
from activation patching). Trainable parameters depend on the condition:
  full       : all blocks
  localized  : layers 3..9
  random     : 7 random layers from 11..30
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint, read_manifest
from evo.tokenizer import CharLevelTokenizer
from phase2.utils import (
    LOCALIZED_LAYERS,
    count_trainable,
    freeze_all,
    iterate_batches,
    select_random_layers,
    set_block_grad,
    tokenize_batch,
)


def filter_train(records):
    return [r for r in records if r.split == "train"]


def configure_trainable_blocks(model, condition: str, seed: int) -> List[int]:
    freeze_all(model)
    if condition == "full":
        layers = list(range(len(model.blocks)))
        for p in model.parameters():
            p.requires_grad_(True)
    elif condition == "localized":
        layers = LOCALIZED_LAYERS
        set_block_grad(model, layers, True)
    elif condition == "random":
        layers = select_random_layers(seed, n=len(LOCALIZED_LAYERS))
        set_block_grad(model, layers, True)
    else:
        raise ValueError(f"Unknown condition {condition}")
    return layers


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
               normalize: bool = False) -> torch.Tensor:
    """MSE between (B, T, D) tensors averaged over valid tokens.

    normalize=True: L2-normalize along D before MSE, making loss scale-invariant.
    This is critical for layers 11+ where activations can reach ~1e15 in bfloat16.
    Normalized MSE is bounded in [0, 4] regardless of activation magnitude.
    """
    a = a.float()
    b = b.float()
    if normalize:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
    diff = (a - b) ** 2          # (B, T, D)
    diff = diff.mean(dim=-1)     # (B, T)
    diff = diff * mask.float()
    denom = mask.float().sum(dim=1).clamp(min=1)
    return (diff.sum(dim=1) / denom).mean()


class HiddenCapture:
    """Capture a single layer's hidden state during forward pass."""
    def __init__(self, model, layer_idx: int):
        self.layer_idx = layer_idx
        self.captured = None
        self.handle = model.blocks[layer_idx].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.captured = hidden  # keep grad graph

    def get(self) -> torch.Tensor:
        return self.captured

    def remove(self) -> None:
        self.handle.remove()


def save_block_deltas(model, layers: List[int], out_path: str) -> None:
    delta = {}
    sd = model.state_dict()
    if len(layers) == len(model.blocks):
        for key, val in sd.items():
            delta[key] = val.detach().to(torch.bfloat16).cpu()
    else:
        for layer_idx in layers:
            prefix = f"blocks.{layer_idx}."
            for key, val in sd.items():
                if key.startswith(prefix):
                    delta[key] = val.detach().to(torch.bfloat16).cpu()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(delta, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget-csv", default="data/phase2/splits/forget.csv")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase2/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=["full", "localized", "random"], required=True)
    parser.add_argument("--target-layer", type=int, default=6,
                        help="Hook layer for representation steering (strongest causal layer).")
    parser.add_argument("--steer-coef", type=float, default=20.0,
                        help="Magnitude of random target direction for forget set.")
    parser.add_argument("--alpha-retain", type=float, default=1.0)
    parser.add_argument("--alpha-forget", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the output directory name (default: rmu_<condition>).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print(f"[RMU] condition={args.condition} target_layer={args.target_layer} c={args.steer_coef}")

    # Load training model
    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.train()
    tokenizer = CharLevelTokenizer(512)
    hidden_dim = model.blocks[0].pre_norm.scale.shape[0]  # 4096

    # Reference model (frozen) for retain MSE
    ref_model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    layers = configure_trainable_blocks(model, args.condition, args.seed)
    print(f"[RMU] trainable layers: {layers}")
    print(f"[RMU] trainable params: {count_trainable(model):,}")

    # The hook layer must be <= the highest trainable layer so gradients can flow back.
    # For localized/full, target_layer=6 is fine (layers 3-9 or all are upstream).
    # For random (layers 11-30), we use the last trainable layer as hook and normalize
    # hidden states to prevent the ~1e15 activation explosion in layers 11+.
    target_layer = args.target_layer
    normalize_hidden = False
    if target_layer not in layers and max(layers) > target_layer:
        target_layer = max(layers)
        normalize_hidden = True
        print(f"[RMU] target_layer {args.target_layer} not in trainable set; "
              f"using layer {target_layer} with normalized MSE (activations unstable in layers 11+)")

    # Hooks at the target layer for both models
    train_hook = HiddenCapture(model, target_layer)
    ref_hook = HiddenCapture(ref_model, target_layer)

    # Fixed random unit direction for the forget set
    g = torch.Generator(device=args.device).manual_seed(args.seed + 999)
    random_dir = torch.randn(hidden_dim, generator=g, device=args.device).float()
    random_dir = random_dir / random_dir.norm().clamp(min=1e-8)
    target_vec = (args.steer_coef * random_dir).to(torch.float32)  # (D,)
    print(f"[RMU] random target dir norm={target_vec.norm().item():.2f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    forget = filter_train(read_manifest(args.forget_csv))
    retain = filter_train(read_manifest(args.retain_csv))
    print(f"[RMU] forget train={len(forget)}  retain train={len(retain)}")

    forget_iter = iter([])
    retain_iter = iter([])

    def next_forget():
        nonlocal forget_iter
        try:
            return next(forget_iter)
        except StopIteration:
            forget_iter = iter(iterate_batches(forget, args.batch_size, shuffle=True, rng=rng))
            return next(forget_iter)

    def next_retain():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(iterate_batches(retain, args.batch_size, shuffle=True, rng=rng))
            return next(retain_iter)

    log_rows = []
    pbar = tqdm(range(args.steps), desc=f"RMU-{args.condition}")
    t0 = time.time()
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        # Forget batch: push hidden at target_layer toward random direction
        fbatch = next_forget()
        f_ids, f_mask = tokenize_batch([r.sequence for r in fbatch], tokenizer, args.max_length, args.device)
        _ = model(f_ids, padding_mask=f_mask)
        f_hidden = train_hook.get()  # (B, T, D) with grad
        target = target_vec.view(1, 1, -1).expand_as(f_hidden)
        L_forget = masked_mse(f_hidden, target, f_mask, normalize=normalize_hidden)

        # Retain batch: keep hidden at target_layer close to reference model
        rbatch = next_retain()
        r_ids, r_mask = tokenize_batch([r.sequence for r in rbatch], tokenizer, args.max_length, args.device)
        with torch.no_grad():
            _ = ref_model(r_ids, padding_mask=r_mask)
            r_hidden_ref = ref_hook.get().detach()
        _ = model(r_ids, padding_mask=r_mask)
        r_hidden = train_hook.get()
        L_retain = masked_mse(r_hidden, r_hidden_ref, r_mask, normalize=normalize_hidden)

        loss = args.alpha_forget * L_forget + args.alpha_retain * L_retain
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            row = {
                "step": step + 1,
                "L_forget_mse": L_forget.item(),
                "L_retain_mse": L_retain.item(),
                "loss": loss.item(),
            }
            log_rows.append(row)
            pbar.set_postfix(Lf=f"{row['L_forget_mse']:.2f}", Lr=f"{row['L_retain_mse']:.2f}")

    elapsed = time.time() - t0
    train_hook.remove()
    ref_hook.remove()
    print(f"[RMU] done in {elapsed:.1f}s")

    run_name = args.run_name if args.run_name else f"rmu_{args.condition}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_block_deltas(model, layers=layers, out_path=os.path.join(run_dir, "weights.safetensors"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "method": "rmu",
            "condition": args.condition,
            "layers": layers,
            "target_layer": target_layer,
            "normalize_hidden": normalize_hidden,
            "steer_coef": args.steer_coef,
            "steps": args.steps, "lr": args.lr,
            "alpha_forget": args.alpha_forget, "alpha_retain": args.alpha_retain,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "seed": args.seed, "elapsed_sec": elapsed,
        }, f, indent=2)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"[RMU] saved to {run_dir}")


if __name__ == "__main__":
    main()
