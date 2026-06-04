"""
Phase 3 — SFT Recovery Attack.

Fine-tune an unlearned checkpoint on held-out viral sequences (test split, label=1)
using standard next-token CE loss. Evaluate probe AUROC before and after attack.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint, read_manifest
from evo.tokenizer import CharLevelTokenizer
from phase2.utils import language_model_loss, tokenize_batch
from phase3.utils import apply_checkpoint, eval_auroc_all_layers, write_results

EVAL_LAYERS = list(range(0, 11))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to unlearned weights.safetensors")
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--probe-dir", default="data/host_tropism/probes")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase3")
    parser.add_argument("--run-name", default=None,
                        help="Override output directory name (default: <ckpt_parent>_sft).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    run_name = os.path.basename(os.path.dirname(args.ckpt))
    out_name = args.run_name if args.run_name else f"{run_name}_sft"
    out_dir = os.path.join(args.out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SFT] attacking {run_name} → {out_dir}")

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    apply_checkpoint(model, args.ckpt)
    tokenizer = CharLevelTokenizer(512)

    # Attack data: val label=1 for fine-tuning; test split (both labels) for eval.
    # Keeping these disjoint ensures recovery AUROC reflects generalisation, not memorisation.
    all_records = read_manifest(args.manifest)
    attack_records = [r for r in all_records if r.split == "val" and r.label == 1]
    eval_records   = [r for r in all_records if r.split == "test"]
    print(f"[SFT] attack-train: {len(attack_records)} sequences (val, label=1)")
    print(f"[SFT] attack-eval:  {len(eval_records)} sequences (test, both labels)")

    # Eval before attack
    model.eval()
    auroc_before = eval_auroc_all_layers(model, eval_records, tokenizer, args.probe_dir,
                                         EVAL_LAYERS, args.batch_size, args.max_length, args.device)
    print("[SFT] AUROC before: " + "  ".join(f"L{l}={v:.3f}" for l, v in auroc_before.items()))

    # Fine-tune on attack set
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

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
    pbar = tqdm(range(args.steps), desc=f"SFT-{run_name}")
    t0 = time.time()
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch = next_batch()
        ids, mask = tokenize_batch([r.sequence for r in batch], tokenizer, args.max_length, args.device)
        logits, _ = model(ids, padding_mask=mask)
        loss = language_model_loss(logits, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            log_rows.append({"step": step + 1, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    elapsed = time.time() - t0

    # Eval after attack
    model.eval()
    auroc_after = eval_auroc_all_layers(model, eval_records, tokenizer, args.probe_dir,
                                        EVAL_LAYERS, args.batch_size, args.max_length, args.device)
    print("[SFT] AUROC after:  " + "  ".join(f"L{l}={v:.3f}" for l, v in auroc_after.items()))

    # Save results
    rows = [{"layer": l, "auroc_before": auroc_before[l],
             "auroc_after": auroc_after[l],
             "auroc_delta": auroc_after[l] - auroc_before[l]}
            for l in EVAL_LAYERS]
    write_results(os.path.join(out_dir, "auroc.csv"), rows)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"run": run_name, "attack": "sft", "steps": args.steps,
                   "lr": args.lr, "elapsed_sec": elapsed,
                   "n_attack_train": len(attack_records),
                   "n_attack_eval": len(eval_records),
                   "attack_split": "val", "eval_split": "test",
                   "lr_grid": True}, f, indent=2)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"[SFT] saved to {out_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
