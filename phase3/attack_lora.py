"""
Phase 3 — LoRA Recovery Attack.

Manually implements LoRA adapters (no peft/transformers dependency) on the
Linear modules inside blocks 3-9 of an unlearned Evo checkpoint.
Fine-tunes on held-out viral sequences (test split, label=1).
Evaluates probe AUROC before and after attack.
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint, read_manifest
from evo.tokenizer import CharLevelTokenizer
from phase2.utils import language_model_loss, tokenize_batch
from phase3.utils import apply_checkpoint, eval_auroc_all_layers, write_results

EVAL_LAYERS = list(range(0, 11))
LORA_TARGET_LAYERS = [3, 4, 5, 6, 7, 8, 9]


class LoRALinear(nn.Module):
    """Wraps an existing frozen Linear with a low-rank adapter A·B."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=linear.weight.device,
                                               dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=linear.weight.device,
                                               dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # lora_B stays zero so initial output is unchanged

    def forward(self, x):
        base = self.linear(x)
        # cast to float32 for adapter computation, then back
        lora_out = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora_out.to(base.dtype)


def inject_lora(model, target_layers: List[int], rank: int, alpha: int) -> List[nn.Parameter]:
    """Replace Linear modules in target blocks with LoRALinear. Returns adapter params."""
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    adapter_params = []
    for layer_idx in target_layers:
        block = model.blocks[layer_idx]
        for attr_path, module in list(block.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # Navigate to parent and replace
            parts = attr_path.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            adapter_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    for p in adapter_params:
        p.requires_grad_(True)
    return adapter_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--probe-dir", default="data/host_tropism/probes")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase3")
    parser.add_argument("--run-name", default=None,
                        help="Override output directory name (default: <ckpt_parent>_lora).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    run_name = os.path.basename(os.path.dirname(args.ckpt))
    out_name = args.run_name if args.run_name else f"{run_name}_lora"
    out_dir = os.path.join(args.out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[LoRA] attacking {run_name} → {out_dir}")

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    apply_checkpoint(model, args.ckpt)
    tokenizer = CharLevelTokenizer(512)

    all_records = read_manifest(args.manifest)
    attack_records = [r for r in all_records if r.split == "val" and r.label == 1]
    eval_records   = [r for r in all_records if r.split == "test"]
    print(f"[LoRA] attack-train: {len(attack_records)} sequences (val, label=1)")
    print(f"[LoRA] attack-eval:  {len(eval_records)} sequences (test, both labels)")

    # Eval before (no LoRA yet)
    model.eval()
    auroc_before = eval_auroc_all_layers(model, eval_records, tokenizer, args.probe_dir,
                                         EVAL_LAYERS, args.batch_size, args.max_length, args.device)
    print("[LoRA] AUROC before: " + "  ".join(f"L{l}={v:.3f}" for l, v in auroc_before.items()))

    # Inject LoRA adapters
    adapter_params = inject_lora(model, LORA_TARGET_LAYERS, args.lora_rank, args.lora_alpha)
    n_adapter = sum(p.numel() for p in adapter_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] adapter params: {n_adapter:,} / {n_total:,} ({100*n_adapter/n_total:.2f}%)")

    model.train()
    optimizer = torch.optim.AdamW(adapter_params, lr=args.lr, weight_decay=0.0)

    def iter_batches():
        indices = list(range(len(attack_records)))
        rng.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            yield [attack_records[i] for i in indices[start: start + args.batch_size]]

    attack_iter = iter([])

    def next_batch():
        nonlocal attack_iter
        try:
            return next(attack_iter)
        except StopIteration:
            attack_iter = iter(iter_batches())
            return next(attack_iter)

    log_rows = []
    pbar = tqdm(range(args.steps), desc=f"LoRA-{run_name}")
    t0 = time.time()
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch = next_batch()
        ids, mask = tokenize_batch([r.sequence for r in batch], tokenizer, args.max_length, args.device)
        logits, _ = model(ids, padding_mask=mask)
        loss = language_model_loss(logits, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            log_rows.append({"step": step + 1, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    elapsed = time.time() - t0

    model.eval()
    auroc_after = eval_auroc_all_layers(model, eval_records, tokenizer, args.probe_dir,
                                        EVAL_LAYERS, args.batch_size, args.max_length, args.device)
    print("[LoRA] AUROC after:  " + "  ".join(f"L{l}={v:.3f}" for l, v in auroc_after.items()))

    rows = [{"layer": l, "auroc_before": auroc_before[l],
             "auroc_after": auroc_after[l],
             "auroc_delta": auroc_after[l] - auroc_before[l]}
            for l in EVAL_LAYERS]
    write_results(os.path.join(out_dir, "auroc.csv"), rows)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"run": run_name, "attack": "lora", "steps": args.steps,
                   "lr": args.lr, "lora_rank": args.lora_rank,
                   "lora_target_layers": LORA_TARGET_LAYERS,
                   "n_adapter_params": n_adapter,
                   "elapsed_sec": elapsed,
                   "n_attack_train": len(attack_records),
                   "n_attack_eval": len(eval_records),
                   "attack_split": "val", "eval_split": "test",
                   "lr_grid": True}, f, indent=2)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"[LoRA] saved to {out_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
