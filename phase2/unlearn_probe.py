"""
Probe-guided Phase 2 unlearning.

This method trains localized layers 5-9 with a representation-level forget
objective. The default zero-logit objective drives each positive target batch
toward the fixed probe decision boundary rather than maximizing the target
score. Retain preservation is anchored against a frozen base model on the
merged Phase 2 retain split.
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
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.checkpoint_io import (
    DEFAULT_TRAINABLE_MODULE_SUFFIXES,
    parse_module_suffixes,
    save_checkpoint,
    select_tensor_names,
    set_trainable_by_suffixes,
    snapshot_state,
)
from phase2.run_metadata import build_run_metadata, write_metadata


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


def filter_train(records, label: int | None = None):
    result = [record for record in records if record.split == "train"]
    if label is not None:
        result = [record for record in result if record.label == label]
    return result


class NextNormCapture:
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = sorted(set(layers))
        self.mask = None
        self.outputs: Dict[int, torch.Tensor] = {}
        self.num_layers = len(model.blocks)
        self.handles = [
            model.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            for layer_idx in self.layers
        ]

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx + 1 < self.num_layers:
                hidden = self.model.blocks[layer_idx + 1].pre_norm(hidden)
            else:
                hidden = self.model.norm(hidden)
            if self.mask is None:
                raise RuntimeError("Mask must be set before running the model.")
            denom = self.mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden * self.mask.unsqueeze(-1).float()).sum(dim=1) / denom
            self.outputs[layer_idx] = pooled

        return hook

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def clear(self) -> None:
        self.outputs = {}

    def get(self, layer_idx: int) -> torch.Tensor:
        return self.outputs[layer_idx]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-target-config", default="phase2/internal_eval_targets.json")
    parser.add_argument("--forget-csv", default="data/phase2/splits/forget.csv", help="Unused; accepted for runner compatibility.")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase2/checkpoints")
    parser.add_argument("--run-name", default="probe_guided_localized")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=["localized"], default="localized")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-forget", type=float, default=1.0)
    parser.add_argument("--alpha-retain", type=float, default=20.0)
    parser.add_argument(
        "--forget-objective",
        choices=["logit_zero", "component_zero"],
        default="logit_zero",
        help=(
            "Probe-guided forget loss. logit_zero minimizes the squared fixed-probe "
            "logit including intercept; component_zero preserves the older squared "
            "standardized component along the normalized probe direction."
        ),
    )
    parser.add_argument("--retain-cosine-weight", type=float, default=0.0)
    parser.add_argument(
        "--retain-loss-components",
        default="hidden_mse,output_kl,ce",
        help="Comma-separated retain losses from hidden_mse,output_kl,ce.",
    )
    parser.add_argument("--forget-probe-config", default="", help="Reserved for multi-probe forget config.")
    parser.add_argument("--capability-probe-config", default="", help="Reserved for capability probe config.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-steps", default="")
    parser.add_argument(
        "--save-policy",
        choices=["full", "selected_modules", "delta", "adapter"],
        default="delta",
    )
    parser.add_argument(
        "--trainable-module-suffixes",
        default=",".join(DEFAULT_TRAINABLE_MODULE_SUFFIXES),
    )
    parser.add_argument("--save-final-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--init-ckpt", default="")
    parser.add_argument("--init-from-run", default="")
    args = parser.parse_args()

    from phase1.utils import load_local_checkpoint, read_manifest
    from evo.tokenizer import CharLevelTokenizer
    from phase2.probe_utils import (
        apply_checkpoint,
        load_probe,
        load_target_specs,
        normalized_standard_probe_direction,
    )
    from phase2.utils import count_trainable, iterate_batches, language_model_loss, tokenize_batch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    save_steps = {step for step in parse_save_steps(args.save_steps) if 1 <= step <= args.steps}
    if args.save_final_only or args.save_best_only:
        save_steps = set()
    retain_loss_components = {part.strip() for part in args.retain_loss_components.split(",") if part.strip()}
    allowed_retain_components = {"hidden_mse", "output_kl", "ce"}
    unknown_retain = retain_loss_components - allowed_retain_components
    if unknown_retain:
        raise ValueError(f"Unknown retain loss components: {sorted(unknown_retain)}")
    if not retain_loss_components:
        raise ValueError("--retain-loss-components must include at least one component")
    trainable_suffixes = parse_module_suffixes(args.trainable_module_suffixes)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    init_ckpt, init_source = resolve_init_ckpt(args)

    target_specs = load_target_specs(args.internal_target_config)
    layers = sorted({layer for spec in target_specs for layer in spec["layers"]})
    if layers != [5, 6, 7, 8, 9]:
        print(f"[probe-guided] warning: target config layers resolved to {layers}")
    tokenizer = CharLevelTokenizer(512)

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if init_ckpt:
        apply_checkpoint(model, init_ckpt)
    model.train()
    trainable_names = set_trainable_by_suffixes(model, layers, trainable_suffixes)
    if args.save_policy == "adapter" and not select_tensor_names(model, layers, trainable_suffixes, adapter_only=True):
        raise ValueError("save-policy=adapter requested, but this model has no adapter/LoRA/IA3 tensors")
    init_state = snapshot_state(model, trainable_names)

    ref_model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    trainable = [param for param in model.parameters() if param.requires_grad]
    print(f"[probe-guided] trainable layers: {layers}")
    print(f"[probe-guided] trainable suffixes: {trainable_suffixes}")
    print(f"[probe-guided] trainable params: {count_trainable(model):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    retain_records = filter_train(read_manifest(args.retain_csv))
    print(f"[probe-guided] retain train={len(retain_records)}")
    if not retain_records:
        raise ValueError(f"No train retain records found in {args.retain_csv}")

    for spec in target_specs:
        spec["train_records"] = filter_train(read_manifest(spec["manifest"]), label=1)
        if not spec["train_records"]:
            raise ValueError(
                f"No positive train records found for target={spec['name']} in {spec['manifest']}"
            )
        spec["probes"] = {
            layer: load_probe(spec["probe_dir"], layer, device=args.device)
            for layer in spec["layers"]
        }
        spec["directions"] = {
            layer: normalized_standard_probe_direction(spec["probes"][layer]).to(args.device)
            for layer in spec["layers"]
        }
        print(
            f"[probe-guided] target={spec['name']} layers={spec['layers']} "
            f"forget_train={len(spec['train_records'])}"
        )

    model_capture = NextNormCapture(model, layers)
    ref_capture = NextNormCapture(ref_model, layers)

    iterators = {spec["name"]: iter([]) for spec in target_specs}
    retain_iter = iter([])

    data_paths = [args.forget_csv, args.retain_csv, args.internal_target_config] + [
        spec["manifest"] for spec in target_specs
    ]
    probe_paths = [
        os.path.join(spec["probe_dir"], f"layer_{layer}.npz")
        for spec in target_specs
        for layer in spec["layers"]
    ]

    def save_model_checkpoint(checkpoint_dir: str, checkpoint_step: int | None, elapsed: float | None):
        out_path = os.path.join(checkpoint_dir, "weights.safetensors")
        save_result = save_checkpoint(
            model,
            out_path,
            policy=args.save_policy,
            layers=layers,
            suffixes=trainable_suffixes,
            init_state=init_state,
            min_free_disk_gb=args.min_free_disk_gb,
            metadata={
                "created_by": "phase2/unlearn_probe.py",
                "source_checkpoint": args.model_dir,
                "init_ckpt": init_ckpt or "",
                "trainable_module_suffixes": ",".join(trainable_suffixes),
                "loss_layers": ",".join(str(layer) for layer in layers),
            },
        )
        metadata = build_run_metadata(
            args=args,
            source_checkpoint=args.model_dir,
            init_checkpoint=init_ckpt or "",
            output_checkpoint=out_path if save_result.saved else "",
            data_paths=data_paths,
            probe_paths=probe_paths,
            trainable_modules=trainable_suffixes,
            trainable_tensor_names=trainable_names,
            trainable_param_count=count_trainable(model),
            loss_layers=layers,
            seed=args.seed,
            save_policy=args.save_policy,
            checkpoint_policy=args.save_policy,
            extra={
                "method": "probe_guided",
                "condition": args.condition,
                "layers": layers,
                "target_names": [spec["name"] for spec in target_specs],
                "internal_target_config": args.internal_target_config,
                "checkpoint_step": checkpoint_step,
                "steps": args.steps,
                "lr": args.lr,
                "alpha_forget": args.alpha_forget,
                "alpha_retain": args.alpha_retain,
                "forget_objective": args.forget_objective,
                "retain_loss_components": sorted(retain_loss_components),
                "retain_cosine_weight": args.retain_cosine_weight,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "forget_csv": args.forget_csv,
                "retain_csv": args.retain_csv,
                "elapsed_sec": elapsed,
                "save_steps": sorted(save_steps),
                "init_source": init_source,
                "init_ckpt": init_ckpt,
                "parent_run": args.run_name if checkpoint_step is not None else "",
                "save_result": save_result.__dict__,
                "save_skipped_reason": save_result.skipped_reason,
            },
        )
        write_metadata(os.path.join(checkpoint_dir, "meta.json"), metadata)
        return save_result

    def next_target_batch(spec: dict):
        nonlocal iterators
        target_name = spec["name"]
        try:
            return next(iterators[target_name])
        except StopIteration:
            iterators[target_name] = iter(
                iterate_batches(spec["train_records"], args.batch_size, shuffle=True, rng=rng)
            )
            return next(iterators[target_name])

    def next_retain_batch():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(iterate_batches(retain_records, args.batch_size, shuffle=True, rng=rng))
            return next(retain_iter)

    log_rows = []
    t0 = time.time()
    pbar = tqdm(range(args.steps), desc="Probe-guided")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        forget_losses = []
        forget_metrics = {}
        for spec in target_specs:
            fbatch = next_target_batch(spec)
            f_ids, f_mask = tokenize_batch(
                [record.sequence for record in fbatch],
                tokenizer,
                args.max_length,
                args.device,
            )
            model_capture.set_mask(f_mask)
            model_capture.clear()
            _ = model(f_ids, padding_mask=f_mask)
            target_layer_losses = []
            layer_metrics = {}
            for layer in spec["layers"]:
                pooled = model_capture.get(layer).float()
                probe = spec["probes"][layer]
                direction = spec["directions"][layer]
                standardized = (pooled - probe["mean"]) / probe["scale"].clamp(min=1e-12)
                logits = (standardized @ probe["coef"].reshape(-1, 1).float()).reshape(-1) + probe["intercept"].reshape(-1)[0].float()
                components = standardized @ direction
                if args.forget_objective == "logit_zero":
                    layer_loss = (logits ** 2).mean()
                else:
                    layer_loss = (components ** 2).mean()
                target_layer_losses.append(layer_loss)
                layer_metrics[str(layer)] = {
                    "fixed_probe_logit_rms": torch.sqrt((logits ** 2).mean()).item(),
                    "probe_component_rms": torch.sqrt((components ** 2).mean()).item(),
                }
            target_loss = torch.stack(target_layer_losses).mean()
            forget_losses.append(target_loss)
            forget_metrics[spec["name"]] = {
                "forget_loss": target_loss.item(),
                "layer_metrics": layer_metrics,
            }

        L_forget = torch.stack(forget_losses).mean()

        rbatch = next_retain_batch()
        r_ids, r_mask = tokenize_batch(
            [record.sequence for record in rbatch],
            tokenizer,
            args.max_length,
            args.device,
        )
        ref_capture.set_mask(r_mask)
        ref_capture.clear()
        with torch.no_grad():
            ref_logits, _ = ref_model(r_ids, padding_mask=r_mask)
        model_capture.set_mask(r_mask)
        model_capture.clear()
        model_logits, _ = model(r_ids, padding_mask=r_mask)

        retain_mse_losses = []
        retain_cosine_losses = []
        retain_layer_metrics = {}
        for layer in layers:
            ref_pooled = ref_capture.get(layer).detach().float()
            pooled = model_capture.get(layer).float()
            mse = ((pooled - ref_pooled) ** 2).mean()
            cosine_penalty = 1.0 - F.cosine_similarity(pooled, ref_pooled, dim=-1).mean()
            retain_mse_losses.append(mse)
            retain_cosine_losses.append(cosine_penalty)
            retain_layer_metrics[str(layer)] = {
                "retain_mse": mse.item(),
                "retain_cosine_penalty": cosine_penalty.item(),
            }

        L_retain_mse = torch.stack(retain_mse_losses).mean()
        L_retain_cosine = torch.stack(retain_cosine_losses).mean()
        retain_components = {}
        if "hidden_mse" in retain_loss_components:
            retain_components["hidden_mse"] = L_retain_mse
        if "output_kl" in retain_loss_components:
            shifted_log_probs = F.log_softmax(model_logits[:, :-1, :].float(), dim=-1)
            shifted_ref_probs = F.softmax(ref_logits[:, :-1, :].detach().float(), dim=-1)
            shifted_mask = r_mask[:, 1:].float()
            token_kl = F.kl_div(shifted_log_probs, shifted_ref_probs, reduction="none").sum(dim=-1)
            retain_components["output_kl"] = (token_kl * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)
        if "ce" in retain_loss_components:
            retain_components["ce"] = language_model_loss(model_logits, r_ids, r_mask)
        L_retain = torch.stack(list(retain_components.values())).mean()
        if args.retain_cosine_weight:
            L_retain = L_retain + args.retain_cosine_weight * L_retain_cosine

        weighted_forget = args.alpha_forget * L_forget
        weighted_retain = args.alpha_retain * L_retain
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
                "L_retain_mse": L_retain_mse.item(),
                "L_retain_cosine": L_retain_cosine.item(),
                "L_retain_output_kl": retain_components.get("output_kl", torch.tensor(float("nan"))).item(),
                "L_retain_ce": retain_components.get("ce", torch.tensor(float("nan"))).item(),
                "L_retain_total": L_retain.item(),
                "weighted_forget_term": weighted_forget.item(),
                "weighted_retain_term": weighted_retain.item(),
                "loss": loss.item(),
                "forget_target_metrics": forget_metrics,
                "retain_layer_metrics": retain_layer_metrics,
            }
            log_rows.append(row)
            pbar.set_postfix(Lf=f"{row['L_forget']:.4f}", Lr=f"{row['L_retain_total']:.4f}")

        if current_step in save_steps:
            step_dir = os.path.join(run_dir, f"step_{current_step:06d}")
            save_model_checkpoint(step_dir, current_step, time.time() - t0)

    elapsed = time.time() - t0
    final_save = save_model_checkpoint(run_dir, None, elapsed)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    model_capture.close()
    ref_capture.close()
    if final_save.saved:
        print(f"[probe-guided] saved to {run_dir} in {elapsed:.1f}s")
    else:
        print(
            f"[probe-guided] checkpoint save skipped for {run_dir}: "
            f"{final_save.skipped_reason} free={final_save.free_disk_gb:.2f}G "
            f"min={final_save.min_free_disk_gb:.2f}G"
        )


if __name__ == "__main__":
    main()
