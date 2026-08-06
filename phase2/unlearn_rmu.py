"""
RMU (Representation Misdirection for Unlearning) for Evo-1-8k-base.

Idea (Li et al., ICML 2024):
  - Forget set: push hidden activations at one or more target layers toward a
    steering direction (random, non-human, or joint probe-derived).
  - Retain set: keep hidden activations close to those of a *frozen* reference
    copy of the model.

This implementation supports multi-layer RMU. That matters for this project:
Phase 1 localizes the merged selective-unlearning signal to a span of layers
rather than a single layer, so a one-layer hook can leave later "localized"
layers with no effective gradient. Trainable parameters depend on the
condition:
  full       : all blocks
  localized  : layers loaded from localized_layers.json
  random     : matched random layers from 11..30
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1 import utils as phase1_utils
from evo.tokenizer import CharLevelTokenizer
from phase2 import utils as phase2_utils
from phase2.probe_utils import (
    load_probe,
    load_target_specs,
    normalized_raw_probe_direction,
    orthonormal_basis,
)
from phase2.run_metadata import build_run_metadata, write_metadata

load_local_checkpoint = phase1_utils.load_local_checkpoint
read_manifest = getattr(phase1_utils, "read_manifest", None)
count_trainable = getattr(phase2_utils, "count_trainable", None)
freeze_all = getattr(phase2_utils, "freeze_all", None)
get_localized_layers = getattr(phase2_utils, "get_localized_layers", None)
get_primary_target_layer = getattr(phase2_utils, "get_primary_target_layer", None)
iterate_batches = getattr(phase2_utils, "iterate_batches", None)
select_random_layers = getattr(phase2_utils, "select_random_layers", None)
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


def masked_component_rms(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reference activation RMS per scalar component over valid tokens."""
    values = hidden.detach().float().pow(2).mean(dim=-1)
    values = values * mask.float()
    return torch.sqrt(values.sum() / mask.float().sum().clamp(min=1)).clamp(min=1e-8)


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


def mean_pool_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    return (hidden * mask.unsqueeze(-1).float()).sum(dim=1) / denom


def resolve_loss_layers(
    spec: str,
    condition: str,
    requested_target_layer: int,
    trainable_layers: List[int],
    localized_layers: List[int],
) -> List[int]:
    spec = spec.strip().lower()
    if spec == "auto":
        if condition == "localized":
            layers = [layer for layer in localized_layers if layer in trainable_layers]
            return layers if layers else [requested_target_layer]
        if condition == "random":
            return list(trainable_layers)
        layers = [layer for layer in localized_layers if layer in trainable_layers]
        return layers if layers else [requested_target_layer]
    if spec == "target":
        return [requested_target_layer]
    if spec == "trainable":
        return list(trainable_layers)
    if spec == "localized":
        layers = [layer for layer in localized_layers if layer in trainable_layers]
        if not layers:
            raise ValueError("No localized layers overlap the trainable set.")
        return layers
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    if not layers:
        raise ValueError(f"Could not parse --loss-layers={spec!r}")
    invalid = [layer for layer in layers if layer not in trainable_layers]
    if invalid:
        raise ValueError(
            f"Loss layers must be trainable. Invalid layers: {invalid}; "
            f"trainable={trainable_layers}"
        )
    return sorted(set(layers))


def compute_nonhuman_directions(
    model,
    forget_records: list,
    retain_records: list,
    tokenizer: "CharLevelTokenizer",
    target_layers: List[int],
    batch_size: int,
    max_length: int,
    device: str,
    max_seqs: int = 500,
    seed: int = 0,
) -> Dict[int, torch.Tensor]:
    """Compute per-layer non-human steering directions.

    direction[layer] = normalize(mean_retain_activation - mean_forget_activation)
    """
    rng_local = random.Random(seed)
    captures = {layer: HiddenCapture(model, layer) for layer in target_layers}

    def mean_pool_records(records: list) -> Dict[int, torch.Tensor]:
        subset = list(records)
        if len(subset) > max_seqs:
            rng_local.shuffle(subset)
            subset = subset[:max_seqs]
        accum = {layer: [] for layer in target_layers}
        with torch.inference_mode():
            for start in range(0, len(subset), batch_size):
                batch = subset[start : start + batch_size]
                ids, mask = tokenize_batch(
                    [r.sequence for r in batch], tokenizer, max_length, device
                )
                _ = model(ids, padding_mask=mask)
                for layer, capture in captures.items():
                    hidden = capture.get().float()
                    pooled = mean_pool_hidden(hidden, mask)
                    accum[layer].append(pooled.cpu())
        return {
            layer: torch.cat(pieces, dim=0).mean(dim=0)
            for layer, pieces in accum.items()
        }

    print(
        f"[RMU] computing non-human directions for layers {target_layers}: "
        f"extracting forget activations (up to {max_seqs} seqs) ..."
    )
    mean_forget = mean_pool_records(forget_records)
    print(
        f"[RMU] computing non-human directions for layers {target_layers}: "
        f"extracting retain activations (up to {max_seqs} seqs) ..."
    )
    mean_retain = mean_pool_records(retain_records)
    for capture in captures.values():
        capture.remove()

    directions = {}
    for layer in target_layers:
        direction = (mean_retain[layer] - mean_forget[layer]).to(device)
        norm = direction.norm().clamp(min=1e-8)
        directions[layer] = direction / norm
        print(f"[RMU] layer {layer} non-human direction raw_norm={norm.item():.4f}")
    return directions


def compute_joint_probe_directions(
    internal_target_config: str,
    target_layers: List[int],
    device: str,
    sign: float = -1.0,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    """Compute per-layer directions from the configured target probes.

    The raw probe direction increases the label=1 logit. The default negative
    sign steers label=1 forget examples away from the joint target subspace.
    """
    if not internal_target_config:
        raise ValueError("--internal-target-config is required for --target-direction=joint_probe")
    target_specs = load_target_specs(internal_target_config)
    target_names = [spec["name"] for spec in target_specs]
    directions: Dict[int, torch.Tensor] = {}
    layer_probe_paths: Dict[str, Dict[str, str]] = {}
    layer_target_names: Dict[str, List[str]] = {}

    for layer in target_layers:
        vectors = []
        probe_paths = {}
        names = []
        for spec in target_specs:
            if layer not in spec["layers"]:
                continue
            probe = load_probe(spec["probe_dir"], layer, device=device)
            vectors.append(normalized_raw_probe_direction(probe))
            probe_paths[spec["name"]] = probe["path"]
            names.append(spec["name"])
        if not vectors:
            raise ValueError(
                f"No target probes from {internal_target_config} cover RMU loss layer {layer}"
            )

        direction = torch.stack(vectors, dim=0).sum(dim=0)
        raw_norm = direction.norm()
        if raw_norm <= 1e-6 and len(vectors) > 1:
            basis = orthonormal_basis(vectors)
            direction = basis.sum(dim=1)
            raw_norm = direction.norm()
        direction = sign * direction
        norm = direction.norm().clamp(min=1e-8)
        directions[layer] = (direction / norm).to(device)
        layer_probe_paths[str(layer)] = probe_paths
        layer_target_names[str(layer)] = names
        print(
            f"[RMU] layer {layer} joint-probe direction targets={names} "
            f"sign={sign:g} raw_norm={raw_norm.item():.4f}"
        )

    return directions, {
        "target_names": target_names,
        "layer_probe_paths": layer_probe_paths,
        "layer_target_names": layer_target_names,
    }


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


def build_rmu_metadata(
    *,
    args,
    layers: List[int],
    loss_layers: List[int],
    requested_target_layer: int,
    normalize_hidden: bool,
    direction_metadata: dict,
    effective_alpha_retain: float,
    requested_alpha_retain: float,
    save_steps: List[int],
    elapsed_sec: float | None = None,
    checkpoint_step: int | None = None,
    parent_run: str | None = None,
) -> dict:
    extra = {
        "method": "rmu",
        "condition": args.condition,
        "layers": layers,
        "target_layer": requested_target_layer,
        "loss_layers": loss_layers,
        "normalize_hidden": normalize_hidden,
        "scale_calibrated": args.scale_calibrated,
        "steer_coef": args.steer_coef,
        "target_direction": args.target_direction,
        "direction_seqs": args.direction_seqs if args.target_direction == "nonhuman" else None,
        "internal_target_config": args.internal_target_config,
        "probe_direction_sign": args.probe_direction_sign if args.target_direction == "joint_probe" else None,
        "direction_metadata": direction_metadata,
        "retain_cosine_weight": args.retain_cosine_weight,
        "steps": args.steps,
        "lr": args.lr,
        "alpha_forget": args.alpha_forget,
        "alpha_retain": effective_alpha_retain,
        "requested_alpha_retain": requested_alpha_retain,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "forget_csv": args.forget_csv,
        "retain_csv": args.retain_csv,
        "seed": args.seed,
        "save_steps": save_steps,
        "localized_layers_path": args.localized_layers_path,
    }
    if elapsed_sec is not None:
        extra["elapsed_sec"] = elapsed_sec
    if checkpoint_step is not None:
        extra["checkpoint_step"] = checkpoint_step
    if parent_run is not None:
        extra["parent_run"] = parent_run
    return build_run_metadata(
        args=args,
        source_checkpoint=args.model_dir,
        data_paths=[args.internal_target_config, args.forget_csv, args.retain_csv, args.localized_layers_path],
        loss_layers=loss_layers,
        seed=args.seed,
        extra=extra,
    )


def main() -> None:
    if read_manifest is None:
        raise ImportError("phase1.utils.read_manifest is required to run unlearn_rmu.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget-csv", default="data/phase2/splits/forget.csv")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument(
        "--internal-target-config",
        default="phase2/internal_eval_targets.json",
        help="Target probe config used by --target-direction=joint_probe.",
    )
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase2/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=["full", "localized", "random"], required=True)
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Hook layer for representation steering (strongest causal layer).")
    parser.add_argument("--steer-coef", type=float, default=50.0,
                        help="Magnitude of random target direction for forget set.")
    parser.add_argument(
        "--scale-calibrated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Calibrate RMU MSE by each layer's reference activation RMS. In this mode "
            "--steer-coef is the target-vector norm divided by the typical reference "
            "hidden-vector norm, enabling comparable searches across shallow/deep layers."
        ),
    )
    parser.add_argument(
        "--alpha-retain",
        type=float,
        default=1.0,
        help="Retain anchor weight. Full-model runs are clamped to at least 10.0 to avoid global representation drift.",
    )
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
        "--loss-layers",
        default="auto",
        help=(
            "Which layers receive RMU losses. "
            "'auto': localized span for localized/full, trainable layers for random. "
            "'target': only --target-layer. "
            "'localized': localized_layers.json span. "
            "'trainable': every trainable layer. "
            "Or pass a comma-separated layer list, e.g. 5,6,7,8,9."
        ),
    )
    parser.add_argument(
        "--target-direction",
        choices=["random", "nonhuman", "joint_probe"],
        default="nonhuman",
        help=(
            "Steering direction for forget set. "
            "'random': fixed random unit vector (control only). "
            "'nonhuman': mean(retain) - mean(forget) at target_layer, "
            "steers human-tropic activations toward the non-human-tropic subspace. "
            "'joint_probe': negative equal-weight sum of configured raw target-probe "
            "directions, steers label=1 forget examples away from the joint target."
        ),
    )
    parser.add_argument(
        "--probe-direction-sign",
        type=float,
        default=-1.0,
        help=(
            "Sign applied to joint_probe directions. Default -1 steers positive "
            "forget examples away from the target-probe logits."
        ),
    )
    parser.add_argument(
        "--direction-seqs",
        type=int,
        default=500,
        help="Max sequences per class used to estimate the non-human direction (--target-direction=nonhuman).",
    )
    parser.add_argument(
        "--retain-cosine-weight",
        type=float,
        default=0.0,
        help=(
            "Optional extra retain penalty: alpha_retain * "
            "(MSE + retain_cosine_weight * (1 - cosine))."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    requested_alpha_retain = args.alpha_retain
    effective_alpha_retain = args.alpha_retain
    if args.condition == "full" and effective_alpha_retain < 10.0:
        effective_alpha_retain = 10.0
        print(
            "[RMU] condition=full: raising alpha_retain "
            f"from {requested_alpha_retain} to {effective_alpha_retain} "
            "because weaker retain anchoring caused full-model drift."
        )
    if args.target_direction == "random":
        print(
            "[RMU] warning: --target-direction=random is kept only as a control. "
            "Use nonhuman or joint_probe for real checkpoint selection."
        )

    requested_target_layer = (
        args.target_layer
        if args.target_layer is not None
        else get_primary_target_layer(args.localized_layers_path)
    )
    print(
        f"[RMU] condition={args.condition} target_layer={requested_target_layer} "
        f"c={args.steer_coef} alpha_retain={effective_alpha_retain}"
    )
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

    localized_layers = get_localized_layers(args.localized_layers_path)
    layers = configure_trainable_blocks(model, args.condition, args.seed, args.localized_layers_path)
    print(f"[RMU] trainable layers: {layers}")
    print(f"[RMU] trainable params: {count_trainable(model):,}")

    loss_layers = resolve_loss_layers(
        args.loss_layers,
        args.condition,
        requested_target_layer=requested_target_layer,
        trainable_layers=layers,
        localized_layers=localized_layers,
    )
    normalize_hidden = any(layer >= 11 for layer in loss_layers) and not args.scale_calibrated
    print(f"[RMU] loss layers: {loss_layers}")
    if normalize_hidden:
        print("[RMU] using normalized hidden losses because loss layers include 11+")
    if args.scale_calibrated:
        print("[RMU] using per-batch, per-layer reference-RMS calibration")

    forget = filter_train(read_manifest(args.forget_csv))
    retain = filter_train(read_manifest(args.retain_csv))
    print(f"[RMU] forget train={len(forget)}  retain train={len(retain)}")

    train_hooks = {layer: HiddenCapture(model, layer) for layer in loss_layers}
    ref_hooks = {layer: HiddenCapture(ref_model, layer) for layer in loss_layers}

    # Steering direction for the forget set
    direction_metadata = {}
    if args.target_direction == "nonhuman":
        ref_model.eval()
        raw_directions = compute_nonhuman_directions(
            model=ref_model,
            forget_records=forget,
            retain_records=retain,
            tokenizer=tokenizer,
            target_layers=loss_layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            max_seqs=args.direction_seqs,
            seed=args.seed,
        )
        target_vecs = {
            layer: (args.steer_coef * raw_directions[layer]).to(torch.float32)
            for layer in loss_layers
        }
        direction_metadata = {"direction_seqs": args.direction_seqs}
        for layer in loss_layers:
            print(f"[RMU] layer {layer} non-human target dir norm={target_vecs[layer].norm().item():.2f}")
    elif args.target_direction == "joint_probe":
        raw_directions, direction_metadata = compute_joint_probe_directions(
            internal_target_config=args.internal_target_config,
            target_layers=loss_layers,
            device=args.device,
            sign=args.probe_direction_sign,
        )
        target_vecs = {
            layer: (args.steer_coef * raw_directions[layer]).to(torch.float32)
            for layer in loss_layers
        }
        for layer in loss_layers:
            print(f"[RMU] layer {layer} joint-probe target dir norm={target_vecs[layer].norm().item():.2f}")
    else:
        target_vecs = {}
        for layer in loss_layers:
            g = torch.Generator(device=args.device).manual_seed(args.seed + 999 + layer)
            random_dir = torch.randn(hidden_dim, generator=g, device=args.device).float()
            random_dir = random_dir / random_dir.norm().clamp(min=1e-8)
            target_vecs[layer] = (args.steer_coef * random_dir).to(torch.float32)
            print(f"[RMU] layer {layer} random target dir norm={target_vecs[layer].norm().item():.2f}")

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

        # Forget batch: push hidden states toward steering directions.
        fbatch = next_forget()
        f_ids, f_mask = tokenize_batch([r.sequence for r in fbatch], tokenizer, args.max_length, args.device)
        with torch.no_grad():
            _ = ref_model(f_ids, padding_mask=f_mask)
        _ = model(f_ids, padding_mask=f_mask)
        forget_losses = []
        forget_original_losses = []
        forget_original_cosines = []
        forget_layer_metrics = {}
        for layer in loss_layers:
            f_hidden_ref = ref_hooks[layer].get().detach()
            f_hidden = train_hooks[layer].get()
            target_vec = target_vecs[layer]
            layer_scale = None
            if args.scale_calibrated:
                component_rms = masked_component_rms(f_hidden_ref, f_mask)
                reference_vector_norm = component_rms * (f_hidden.shape[-1] ** 0.5)
                target_vec = target_vec / target_vec.norm().clamp(min=1e-8)
                target_vec = target_vec * args.steer_coef * reference_vector_norm
                layer_scale = component_rms
            target = target_vec.view(1, 1, -1).expand_as(f_hidden)
            if layer_scale is None:
                layer_forget = masked_mse(f_hidden, target, f_mask, normalize=normalize_hidden)
            else:
                layer_forget = masked_mse(
                    f_hidden / layer_scale, target / layer_scale, f_mask, normalize=False
                )
            if layer_scale is None:
                layer_forget_original = masked_mse(
                    f_hidden.detach(), f_hidden_ref, f_mask, normalize=normalize_hidden
                )
            else:
                layer_forget_original = masked_mse(
                    f_hidden.detach() / layer_scale,
                    f_hidden_ref / layer_scale,
                    f_mask,
                    normalize=False,
                )
            layer_forget_cosine = masked_cosine(f_hidden.detach(), f_hidden_ref, f_mask)
            forget_losses.append(layer_forget)
            forget_original_losses.append(layer_forget_original)
            forget_original_cosines.append(layer_forget_cosine)
            forget_layer_metrics[str(layer)] = {
                "forget_to_target_mse": layer_forget.item(),
                "forget_to_original_mse": layer_forget_original.item(),
                "forget_original_modified_cosine": layer_forget_cosine.item(),
            }
        L_forget = torch.stack(forget_losses).mean()
        L_forget_original = torch.stack(forget_original_losses).mean()
        forget_original_cosine = torch.stack(forget_original_cosines).mean()

        # Retain batch: anchor hidden states to the frozen reference.
        rbatch = next_retain()
        r_ids, r_mask = tokenize_batch([r.sequence for r in rbatch], tokenizer, args.max_length, args.device)
        with torch.no_grad():
            _ = ref_model(r_ids, padding_mask=r_mask)
        _ = model(r_ids, padding_mask=r_mask)
        retain_mse_losses = []
        retain_cosine_penalties = []
        retain_layer_metrics = {}
        for layer in loss_layers:
            r_hidden_ref = ref_hooks[layer].get().detach()
            r_hidden = train_hooks[layer].get()
            if args.scale_calibrated:
                retain_scale = masked_component_rms(r_hidden_ref, r_mask)
                retain_mse = masked_mse(
                    r_hidden / retain_scale,
                    r_hidden_ref / retain_scale,
                    r_mask,
                    normalize=False,
                )
            else:
                retain_mse = masked_mse(
                    r_hidden, r_hidden_ref, r_mask, normalize=normalize_hidden
                )
            retain_cosine = 1.0 - masked_cosine(r_hidden, r_hidden_ref, r_mask)
            retain_mse_losses.append(retain_mse)
            retain_cosine_penalties.append(retain_cosine)
            retain_layer_metrics[str(layer)] = {
                "retain_rep_mse": retain_mse.item(),
                "retain_cosine_penalty": retain_cosine.item(),
            }
        L_retain_mse = torch.stack(retain_mse_losses).mean()
        L_retain_cosine = torch.stack(retain_cosine_penalties).mean()
        L_retain = L_retain_mse + (args.retain_cosine_weight * L_retain_cosine)

        weighted_forget = args.alpha_forget * L_forget
        weighted_retain = effective_alpha_retain * L_retain
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
                "L_retain_mse": L_retain_mse.item(),
                "L_retain_cosine": L_retain_cosine.item(),
                "L_retain_total": L_retain.item(),
                "forget_to_target_mse": L_forget.item(),
                "forget_to_original_mse": L_forget_original.item(),
                "retain_rep_mse": L_retain_mse.item(),
                "forget_original_modified_cosine": forget_original_cosine.item(),
                "weighted_forget_term": weighted_forget.item(),
                "weighted_retain_term": weighted_retain.item(),
                "loss": loss.item(),
                "target_norm_mean": float(
                    torch.stack([target_vecs[layer].norm() for layer in loss_layers]).mean().item()
                ),
                "target_variance_mean": float(
                    torch.stack(
                        [target_vecs[layer].float().var(unbiased=False) for layer in loss_layers]
                    ).mean().item()
                ),
                "loss_layers": loss_layers,
                "forget_layer_metrics": forget_layer_metrics,
                "retain_layer_metrics": retain_layer_metrics,
            }
            log_rows.append(row)
            pbar.set_postfix(
                Lf=f"{row['L_forget_mse']:.2f}",
                Lr=f"{row['L_retain_total']:.2f}",
                nL=len(loss_layers),
            )
        if current_step in save_steps:
            step_dir = os.path.join(run_dir, f"step_{current_step:06d}")
            save_block_deltas(model, layers=layers, out_path=os.path.join(step_dir, "weights.safetensors"))
            write_metadata(
                os.path.join(step_dir, "meta.json"),
                build_rmu_metadata(
                    args=args,
                    layers=layers,
                    loss_layers=loss_layers,
                    requested_target_layer=requested_target_layer,
                    normalize_hidden=normalize_hidden,
                    direction_metadata=direction_metadata,
                    effective_alpha_retain=effective_alpha_retain,
                    requested_alpha_retain=requested_alpha_retain,
                    save_steps=sorted(save_steps),
                    checkpoint_step=current_step,
                    parent_run=run_name,
                ),
            )

    elapsed = time.time() - t0
    for hook in train_hooks.values():
        hook.remove()
    for hook in ref_hooks.values():
        hook.remove()
    print(f"[RMU] done in {elapsed:.1f}s")

    save_block_deltas(model, layers=layers, out_path=os.path.join(run_dir, "weights.safetensors"))
    write_metadata(
        os.path.join(run_dir, "meta.json"),
        build_rmu_metadata(
            args=args,
            layers=layers,
            loss_layers=loss_layers,
            requested_target_layer=requested_target_layer,
            normalize_hidden=normalize_hidden,
            direction_metadata=direction_metadata,
            effective_alpha_retain=effective_alpha_retain,
            requested_alpha_retain=requested_alpha_retain,
            save_steps=sorted(save_steps),
            elapsed_sec=elapsed,
        ),
    )
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"[RMU] saved to {run_dir}")


if __name__ == "__main__":
    main()
