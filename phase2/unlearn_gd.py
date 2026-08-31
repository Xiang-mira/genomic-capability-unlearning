"""
Gradient Difference (GD) unlearning for Evo-1-8k-base.

The classic formulation: ascend the language-modelling loss on the forget set
while descending it on the retain set.

Per step:
  L_forget = next-token cross-entropy on a forget batch
  L_retain = next-token cross-entropy on a retain batch
  loss     = -alpha_forget * L_forget + alpha_retain * L_retain
             (maximize forget loss, minimize retain loss)

Condition controls which parameter groups receive grad:
  full       : entire model
  localized  : layers loaded from localized_layers.json (causal layers from patching)
  probe      : probe-visible layers 0..10
  random     : matched random blocks sampled from probe-visible layers

This objective produced every archived GD result in the repository that has a
committed meta.json (`lora_gd_*` in data/phase2/checkpoints_lora_grid/, and
`gd_full_ar5`), including the headline 44-task benchmark rows. It was restored
here after a period in which this filename instead contained a probe-guided
representation objective; that objective now lives in
`phase2/unlearn_probe_repr.py`. See docs/RESULTS.md for the reproducibility
consequences.

NOTE ON DIVERGENCE. `-alpha_forget * L_forget` is unbounded below, so this
objective diverges if run too long or too hot: the archived runs pushed retain
perplexity from ~4.2 to 15.7 (localized) and 37.9 (full). That is inherent to
the formulation, not a bug, and it is why GD trades forgetting against general
capability roughly one-for-one in docs/RESULTS.md. `--forget-loss-cap` is
available to bound the forget term; it defaults to 0.0 (disabled) so the
archived behaviour is reproduced exactly.
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

from phase1 import utils as phase1_utils
from evo.tokenizer import CharLevelTokenizer
from phase2.probe_utils import apply_checkpoint
from phase2.run_metadata import build_run_metadata, write_metadata
from phase2 import utils as phase2_utils

load_local_checkpoint = phase1_utils.load_local_checkpoint
read_manifest = getattr(phase1_utils, "read_manifest", None)
PROBE_LAYERS = getattr(phase2_utils, "PROBE_LAYERS", list(range(11)))
count_trainable = getattr(phase2_utils, "count_trainable", None)
freeze_all = getattr(phase2_utils, "freeze_all", None)
get_localized_layers = getattr(phase2_utils, "get_localized_layers", None)
iterate_batches = getattr(phase2_utils, "iterate_batches", None)
language_model_loss = getattr(phase2_utils, "language_model_loss", None)
set_block_grad = getattr(phase2_utils, "set_block_grad", None)
tokenize_batch = getattr(phase2_utils, "tokenize_batch", None)


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
        rng = random.Random(seed)
        layers = sorted(rng.sample(PROBE_LAYERS, k=min(len(localized_layers), len(PROBE_LAYERS))))
        set_block_grad(model, layers, True)
    else:
        raise ValueError(f"Unknown condition {condition}")
    return layers


def resolve_init_ckpt(args) -> tuple[str | None, str]:
    if args.init_ckpt and args.init_from_run:
        raise ValueError("Pass at most one of --init-ckpt or --init-from-run.")
    if args.init_from_run:
        path = os.path.join(args.out_dir, args.init_from_run, "weights.safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Initial checkpoint from run {args.init_from_run!r} not found: {path}")
        return path, f"run:{args.init_from_run}"
    if args.init_ckpt:
        if not os.path.exists(args.init_ckpt):
            raise FileNotFoundError(f"Initial checkpoint not found: {args.init_ckpt}")
        return args.init_ckpt, f"checkpoint:{args.init_ckpt}"
    return None, "base_model"


def save_block_deltas(model, layers: List[int], out_path: str) -> None:
    """Write the trained blocks' absolute weights (whole state dict when full).

    Kept byte-compatible with the archived GD checkpoints. New methods should
    prefer `phase2.checkpoint_io.save_checkpoint`, which additionally records a
    `checkpoint_policy` and gates on free disk.
    """
    tensors = {}
    state = model.state_dict()
    if len(layers) == len(model.blocks):
        for key, val in state.items():
            tensors[key] = val.detach().to(torch.bfloat16).cpu()
    else:
        for layer_idx in layers:
            prefix = f"blocks.{layer_idx}."
            for key, val in state.items():
                if key.startswith(prefix):
                    tensors[key] = val.detach().to(torch.bfloat16).cpu()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(tensors, out_path)


def gd_loss_terms(
    l_forget: torch.Tensor,
    l_retain: torch.Tensor,
    *,
    alpha_forget: float,
    alpha_retain: float,
    forget_loss_cap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the gradient-difference objective.

    Returns `(effective_forget_loss, weighted_forget_term, weighted_retain_term)`.
    The total objective is the sum of the two weighted terms.

    The forget term is *negated*: gradient difference ascends the forget loss.
    That makes the objective unbounded below, so `forget_loss_cap > 0` clamps the
    forget cross-entropy before weighting. A cap of 0.0 disables clamping and
    reproduces the archived runs exactly.
    """
    effective_forget = l_forget
    if forget_loss_cap > 0.0:
        effective_forget = torch.clamp(l_forget, max=forget_loss_cap)
    return (
        effective_forget,
        -alpha_forget * effective_forget,
        alpha_retain * l_retain,
    )


def build_gd_metadata(
    *,
    args,
    layers: List[int],
    trainable_param_count: int,
    init_source: str,
    init_ckpt: str | None,
    save_steps: List[int],
    elapsed_sec: float | None = None,
    checkpoint_step: int | None = None,
    parent_run: str | None = None,
) -> dict:
    extra = {
        "method": "gradient_difference",
        "loss_type": "cross_entropy_gradient_difference",
        "objective": "-alpha_forget * CE(forget) + alpha_retain * CE(retain)",
        "condition": args.condition,
        "layers": layers,
        "steps": args.steps,
        "lr": args.lr,
        "alpha_forget": args.alpha_forget,
        "alpha_retain": args.alpha_retain,
        "forget_loss_cap": args.forget_loss_cap,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "forget_csv": args.forget_csv,
        "retain_csv": args.retain_csv,
        "seed": args.seed,
        "save_steps": save_steps,
        "localized_layers_path": args.localized_layers_path,
        "init_source": init_source,
        "init_ckpt": init_ckpt,
    }
    if elapsed_sec is not None:
        extra["elapsed_sec"] = elapsed_sec
    if checkpoint_step is not None:
        extra["checkpoint_step"] = checkpoint_step
    if parent_run is not None:
        extra["parent_run"] = parent_run
    return build_run_metadata(
        args=args,
        source_checkpoint=init_ckpt or args.model_dir,
        init_checkpoint=init_ckpt or "",
        data_paths=[args.forget_csv, args.retain_csv, args.localized_layers_path],
        trainable_param_count=trainable_param_count,
        seed=args.seed,
        extra=extra,
    )



def main() -> None:
    if read_manifest is None:
        raise ImportError("phase1.utils.read_manifest is required to run unlearn_gd.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--forget-csv", default="data/phase2/splits/forget.csv")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
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
    parser.add_argument(
        "--forget-loss-cap",
        type=float,
        default=0.0,
        help=(
            "Optional upper bound on the forget cross-entropy before weighting. "
            "0.0 disables it and reproduces the archived runs. Set it to bound "
            "the unbounded ascent term when GD diverges."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-steps",
        default="",
        help="Comma-separated intermediate steps to checkpoint, e.g. 100,200,500,1000.",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Checkpoint directory name; defaults to gd_<condition>.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--init-ckpt", default="")
    parser.add_argument("--init-from-run", default="",
                        help="Initialize from <out-dir>/<run>/weights.safetensors, e.g. a projection baseline.")
    parser.add_argument(
        "--localized-layers-path",
        default="data/family_targets/coronaviridae/localized_layers.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print(f"[GD] condition={args.condition} steps={args.steps} lr={args.lr}")
    print("[GD] loss=-alpha_forget*CE(forget) + alpha_retain*CE(retain)")
    save_steps = {step for step in parse_save_steps(args.save_steps) if 1 <= step <= args.steps}
    run_name = args.run_name if args.run_name else f"gd_{args.condition}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    init_ckpt, init_source = resolve_init_ckpt(args)
    if save_steps:
        print(f"[GD] intermediate save steps: {sorted(save_steps)}")

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if init_ckpt:
        apply_checkpoint(model, init_ckpt)
        print(f"[GD] initialized from {init_source}")
    model.train()
    tokenizer = CharLevelTokenizer(512)

    layers = configure_trainable_blocks(model, args.condition, args.seed, args.localized_layers_path)
    print(f"[GD] trainable layers: {layers}")
    trainable_param_count = count_trainable(model)
    print(f"[GD] trainable params: {trainable_param_count:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    forget = filter_train(read_manifest(args.forget_csv))
    retain = filter_train(read_manifest(args.retain_csv))
    if not forget:
        raise ValueError(f"No train rows in {args.forget_csv}")
    if not retain:
        raise ValueError(f"No train rows in {args.retain_csv}")
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

        # Forget loss: ascend.
        fbatch = next_forget_batch()
        f_ids, f_mask = tokenize_batch([r.sequence for r in fbatch], tokenizer, args.max_length, args.device)
        f_logits, _ = model(f_ids, padding_mask=f_mask)
        L_forget = language_model_loss(f_logits, f_ids, f_mask)

        # Retain loss: descend.
        rbatch = next_retain_batch()
        r_ids, r_mask = tokenize_batch([r.sequence for r in rbatch], tokenizer, args.max_length, args.device)
        r_logits, _ = model(r_ids, padding_mask=r_mask)
        L_retain = language_model_loss(r_logits, r_ids, r_mask)

        L_forget_effective, weighted_forget, weighted_retain = gd_loss_terms(
            L_forget,
            L_retain,
            alpha_forget=args.alpha_forget,
            alpha_retain=args.alpha_retain,
            forget_loss_cap=args.forget_loss_cap,
        )
        loss = weighted_forget + weighted_retain
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        current_step = step + 1
        if current_step % args.log_every == 0 or step == 0:
            row = {
                "step": current_step,
                "L_forget": L_forget.item(),
                "L_forget_effective": L_forget_effective.item(),
                "L_retain": L_retain.item(),
                "weighted_forget_term": weighted_forget.item(),
                "weighted_retain_term": weighted_retain.item(),
                "loss": loss.item(),
            }
            log_rows.append(row)
            pbar.set_postfix(Lf=f"{row['L_forget']:.3f}", Lr=f"{row['L_retain']:.3f}")
        if current_step in save_steps:
            step_dir = os.path.join(run_dir, f"step_{current_step:06d}")
            save_block_deltas(model, layers, os.path.join(step_dir, "weights.safetensors"))
            write_metadata(
                os.path.join(step_dir, "meta.json"),
                build_gd_metadata(
                    args=args,
                    layers=layers,
                    trainable_param_count=trainable_param_count,
                    init_source=init_source,
                    init_ckpt=init_ckpt,
                    save_steps=sorted(save_steps),
                    checkpoint_step=current_step,
                    parent_run=run_name,
                ),
            )

    elapsed = time.time() - t0
    print(f"[GD] done in {elapsed:.1f}s")

    save_block_deltas(model, layers, os.path.join(run_dir, "weights.safetensors"))
    write_metadata(
        os.path.join(run_dir, "meta.json"),
        build_gd_metadata(
            args=args,
            layers=layers,
            trainable_param_count=trainable_param_count,
            init_source=init_source,
            init_ckpt=init_ckpt,
            save_steps=sorted(save_steps),
            elapsed_sec=elapsed,
        ),
    )
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"[GD] saved to {run_dir}")


if __name__ == "__main__":
    main()
