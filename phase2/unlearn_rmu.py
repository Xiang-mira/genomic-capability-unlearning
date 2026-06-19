"""
RMU (Representation Misdirection for Unlearning) for Evo-1-8k-base.

Idea (Li et al., ICML 2024):
  - Forget set: push hidden activations at a target layer toward a random unit
    direction (steering coefficient c). This destroys the structured representation.
  - Retain set: keep hidden activations close to those of a *frozen* reference
    copy of the model.

We use a single target hook layer (default: the strongest causal layer from the
localized_layers.json selection). Trainable parameters depend on the condition:
  full       : all blocks
  localized  : layers loaded from localized_layers.json
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
    count_trainable,
    freeze_all,
    get_localized_layers,
    get_primary_target_layer,
    iterate_batches,
    select_random_layers,
    set_block_grad,
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
    elif condition == "random":
        layers = select_random_layers(seed, n=len(localized_layers))
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


def masked_cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1)
    cos = cos * mask.float()
    denom = mask.float().sum(dim=1).clamp(min=1)
    return (cos.sum(dim=1) / denom).mean()


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


def compute_nonhuman_direction(
    model,
    forget_records: list,
    retain_records: list,
    tokenizer: "CharLevelTokenizer",
    target_layer: int,
    batch_size: int,
    max_length: int,
    device: str,
    max_seqs: int = 500,
    seed: int = 0,
) -> torch.Tensor:
    """Compute the non-human-tropic steering direction at target_layer.

    direction = normalize(mean_retain_activation - mean_forget_activation)

    This is more targeted than a random vector: it steers human-tropic
    representations toward the subspace occupied by non-human-tropic sequences
    rather than toward an arbitrary random direction.

    Uses mean-pooled hidden states (same convention as probe extraction).
    Runs under inference_mode; does not affect model gradients.
    """
    rng_local = random.Random(seed)

    capture = HiddenCapture(model, target_layer)

    def mean_pool_records(records: list) -> torch.Tensor:
        subset = list(records)
        if len(subset) > max_seqs:
            rng_local.shuffle(subset)
            subset = subset[:max_seqs]
        accum = []
        with torch.inference_mode():
            for start in range(0, len(subset), batch_size):
                batch = subset[start : start + batch_size]
                ids, mask = tokenize_batch(
                    [r.sequence for r in batch], tokenizer, max_length, device
                )
                _ = model(ids, padding_mask=mask)
                hidden = capture.get().float()          # (B, T, D)
                denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (hidden * mask.unsqueeze(-1).float()).sum(dim=1) / denom
                accum.append(pooled.cpu())
        return torch.cat(accum, dim=0).mean(dim=0)     # (D,)

    print(f"[RMU] computing non-human direction: extracting forget activations "
          f"(up to {max_seqs} seqs) ...")
    mean_forget = mean_pool_records(forget_records)
    print(f"[RMU] computing non-human direction: extracting retain activations "
          f"(up to {max_seqs} seqs) ...")
    mean_retain = mean_pool_records(retain_records)
    capture.remove()

    direction = (mean_retain - mean_forget).to(device)
    norm = direction.norm().clamp(min=1e-8)
    direction = direction / norm
    print(f"[RMU] non-human direction computed: raw_norm={norm.item():.4f}")
    return direction


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
    parser.add_argument("--forget-csv", default="data/phase2/coronaviridae_splits/forget.csv")
    parser.add_argument("--retain-csv", default="data/phase2/coronaviridae_splits/retain.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase2/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=["full", "localized", "random"], required=True)
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Hook layer for representation steering (strongest causal layer).")
    parser.add_argument("--steer-coef", type=float, default=50.0,
                        help="Magnitude of random target direction for forget set.")
    parser.add_argument("--alpha-retain", type=float, default=1.0)
    parser.add_argument("--alpha-forget", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-steps",
        default="100,200,500,1000",
        help="Comma-separated training steps to save as intermediate checkpoints, e.g. 100,200,500,1000.",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the output directory name (default: rmu_<condition>).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--localized-layers-path",
        default="data/family_targets/coronaviridae/localized_layers.json",
    )
    parser.add_argument(
        "--target-direction",
        choices=["random", "nonhuman"],
        default="random",
        help=(
            "Steering direction for forget set. "
            "'random': fixed random unit vector (original). "
            "'nonhuman': mean(retain) - mean(forget) at target_layer, "
            "steers human-tropic activations toward the non-human-tropic subspace."
        ),
    )
    parser.add_argument(
        "--direction-seqs",
        type=int,
        default=500,
        help="Max sequences per class used to estimate the non-human direction (--target-direction=nonhuman).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    requested_target_layer = (
        args.target_layer
        if args.target_layer is not None
        else get_primary_target_layer(args.localized_layers_path)
    )
    print(f"[RMU] condition={args.condition} target_layer={requested_target_layer} c={args.steer_coef}")
    save_steps = {step for step in parse_save_steps(args.save_steps) if 1 <= step <= args.steps}
    run_name = args.run_name if args.run_name else f"rmu_{args.condition}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if save_steps:
        print(f"[RMU] intermediate save steps: {sorted(save_steps)}")

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

    layers = configure_trainable_blocks(model, args.condition, args.seed, args.localized_layers_path)
    print(f"[RMU] trainable layers: {layers}")
    print(f"[RMU] trainable params: {count_trainable(model):,}")

    # The hook layer must be <= the highest trainable layer so gradients can flow back.
    # For localized/full, target_layer=6 is fine (layers 3-9 or all are upstream).
    # For random (layers 11-30), we use the last trainable layer as hook and normalize
    # hidden states to prevent the ~1e15 activation explosion in layers 11+.
    target_layer = requested_target_layer
    normalize_hidden = False
    if target_layer not in layers and max(layers) > target_layer:
        target_layer = max(layers)
        normalize_hidden = True
        print(f"[RMU] target_layer {requested_target_layer} not in trainable set; "
              f"using layer {target_layer} with normalized MSE (activations unstable in layers 11+)")

    forget = filter_train(read_manifest(args.forget_csv))
    retain = filter_train(read_manifest(args.retain_csv))
    print(f"[RMU] forget train={len(forget)}  retain train={len(retain)}")

    # Hooks at the target layer for both models
    train_hook = HiddenCapture(model, target_layer)
    ref_hook = HiddenCapture(ref_model, target_layer)

    # Steering direction for the forget set
    if args.target_direction == "nonhuman":
        # Computed from ref_model (frozen base) so training model state is unaffected.
        ref_model.eval()
        raw_direction = compute_nonhuman_direction(
            model=ref_model,
            forget_records=forget,
            retain_records=retain,
            tokenizer=tokenizer,
            target_layer=target_layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            max_seqs=args.direction_seqs,
            seed=args.seed,
        )
        target_vec = (args.steer_coef * raw_direction).to(torch.float32)
        print(f"[RMU] non-human target dir norm={target_vec.norm().item():.2f}")
    else:
        g = torch.Generator(device=args.device).manual_seed(args.seed + 999)
        random_dir = torch.randn(hidden_dim, generator=g, device=args.device).float()
        random_dir = random_dir / random_dir.norm().clamp(min=1e-8)
        target_vec = (args.steer_coef * random_dir).to(torch.float32)  # (D,)
        print(f"[RMU] random target dir norm={target_vec.norm().item():.2f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

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
        with torch.no_grad():
            _ = ref_model(f_ids, padding_mask=f_mask)
            f_hidden_ref = ref_hook.get().detach()
        _ = model(f_ids, padding_mask=f_mask)
        f_hidden = train_hook.get()  # (B, T, D) with grad
        target = target_vec.view(1, 1, -1).expand_as(f_hidden)
        L_forget = masked_mse(f_hidden, target, f_mask, normalize=normalize_hidden)
        L_forget_original = masked_mse(f_hidden.detach(), f_hidden_ref, f_mask, normalize=normalize_hidden)
        forget_original_cosine = masked_cosine(f_hidden.detach(), f_hidden_ref, f_mask)

        # Retain batch: keep hidden at target_layer close to reference model
        rbatch = next_retain()
        r_ids, r_mask = tokenize_batch([r.sequence for r in rbatch], tokenizer, args.max_length, args.device)
        with torch.no_grad():
            _ = ref_model(r_ids, padding_mask=r_mask)
            r_hidden_ref = ref_hook.get().detach()
        _ = model(r_ids, padding_mask=r_mask)
        r_hidden = train_hook.get()
        L_retain = masked_mse(r_hidden, r_hidden_ref, r_mask, normalize=normalize_hidden)

        weighted_forget = args.alpha_forget * L_forget
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
                "L_forget_mse": L_forget.item(),
                "L_retain_mse": L_retain.item(),
                "forget_to_target_mse": L_forget.item(),
                "forget_to_original_mse": L_forget_original.item(),
                "retain_rep_mse": L_retain.item(),
                "forget_original_modified_cosine": forget_original_cosine.item(),
                "weighted_forget_term": weighted_forget.item(),
                "weighted_retain_term": weighted_retain.item(),
                "loss": loss.item(),
                "target_norm": target_vec.norm().item(),
                "target_variance": target_vec.float().var(unbiased=False).item(),
            }
            log_rows.append(row)
            pbar.set_postfix(Lf=f"{row['L_forget_mse']:.2f}", Lr=f"{row['L_retain_mse']:.2f}")
        if current_step in save_steps:
            step_dir = os.path.join(run_dir, f"step_{current_step:06d}")
            save_block_deltas(model, layers=layers, out_path=os.path.join(step_dir, "weights.safetensors"))
            with open(os.path.join(step_dir, "meta.json"), "w") as f:
                json.dump({
                    "method": "rmu",
                    "condition": args.condition,
                    "layers": layers,
                    "checkpoint_step": current_step,
                    "target_layer": target_layer,
                    "normalize_hidden": normalize_hidden,
                    "steer_coef": args.steer_coef,
                    "target_direction": args.target_direction,
                    "direction_seqs": args.direction_seqs if args.target_direction == "nonhuman" else None,
                    "steps": args.steps, "lr": args.lr,
                    "alpha_forget": args.alpha_forget, "alpha_retain": args.alpha_retain,
                    "batch_size": args.batch_size, "max_length": args.max_length,
                    "forget_csv": args.forget_csv,
                    "retain_csv": args.retain_csv,
                    "seed": args.seed,
                    "localized_layers_path": args.localized_layers_path,
                    "parent_run": run_name,
                }, f, indent=2)

    elapsed = time.time() - t0
    train_hook.remove()
    ref_hook.remove()
    print(f"[RMU] done in {elapsed:.1f}s")

    save_block_deltas(model, layers=layers, out_path=os.path.join(run_dir, "weights.safetensors"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "method": "rmu",
            "condition": args.condition,
            "layers": layers,
            "target_layer": target_layer,
            "normalize_hidden": normalize_hidden,
            "steer_coef": args.steer_coef,
            "target_direction": args.target_direction,
            "direction_seqs": args.direction_seqs if args.target_direction == "nonhuman" else None,
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
    print(f"[RMU] saved to {run_dir}")


if __name__ == "__main__":
    main()
