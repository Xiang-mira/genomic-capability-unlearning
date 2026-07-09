"""
Apply a joint probe null-space projection to localized residual-writer modules.

This is a training-free baseline: for each localized layer, gather the probe
directions for every configured internal target, orthonormalize them, and
project residual-writer outputs into the complementary subspace.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import load_local_checkpoint
from phase2.probe_utils import (
    load_probe,
    load_target_specs,
    normalized_raw_probe_direction,
    orthonormal_basis,
    parse_layers,
    projection_matrix,
)


ALLOWED_RESIDUAL_WRITER_SUFFIXES = (
    "out_filter_dense",
    "mlp.l3",
    "inner_mha_cls.out_proj",
)

MODULE_SCOPE_SUFFIXES = {
    "all": ALLOWED_RESIDUAL_WRITER_SUFFIXES,
    "mlp_l3": ("mlp.l3",),
    "out_filter_dense": ("out_filter_dense",),
    "attention_out": ("inner_mha_cls.out_proj",),
    "mlp_and_filter": ("mlp.l3", "out_filter_dense"),
}


def parse_target_map(spec: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not spec:
        return result
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected name=value in {spec!r}, got {item!r}")
        name, value = item.split("=", 1)
        result[name.strip()] = value.strip()
    return result


def parse_target_strengths(spec: str) -> Dict[str, float]:
    return {name: float(value) for name, value in parse_target_map(spec).items()}


def apply_layer_overrides(target_specs: List[dict], layers: str, target_layers: str) -> List[dict]:
    global_layers = None
    if layers:
        global_layers = set(parse_layers(layers))
    per_target_layers = {
        name: set(parse_layers(layer_spec))
        for name, layer_spec in parse_target_map(target_layers).items()
    }
    updated = []
    for spec in target_specs:
        selected = set(spec["layers"])
        if global_layers is not None:
            selected &= global_layers
        if spec["name"] in per_target_layers:
            selected = set(per_target_layers[spec["name"]])
        if not selected:
            continue
        spec = dict(spec)
        spec["layers"] = sorted(selected)
        spec["localized_layers"] = [
            layer for layer in spec.get("localized_layers", []) if layer in selected
        ]
        updated.append(spec)
    if not updated:
        raise ValueError("Layer overrides removed every projection target.")
    return updated


def discover_projection_modules(model, layers: List[int], suffixes: Tuple[str, ...]) -> Dict[int, List[dict]]:
    modules_by_layer: Dict[int, List[dict]] = {layer: [] for layer in layers}
    hidden_dim = model.blocks[0].pre_norm.scale.shape[0]
    for layer_idx in layers:
        block = model.blocks[layer_idx]
        for module_path, module in block.named_modules():
            if not module_path or not isinstance(module, nn.Linear):
                continue
            if module.out_features != hidden_dim:
                continue
            if not any(module_path.endswith(suffix) for suffix in suffixes):
                continue
            full_name = f"blocks.{layer_idx}.{module_path}"
            modules_by_layer[layer_idx].append(
                {
                    "full_name": full_name,
                    "module_path": module_path,
                    "module": module,
                }
            )
    return modules_by_layer


def soft_projection_matrix(
    vectors: List[Tuple[str, torch.Tensor]],
    projection_strength: float,
    target_strengths: Dict[str, float],
    hidden_dim: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    eye = torch.eye(hidden_dim, device=device, dtype=torch.float32)
    basis_vectors = []
    basis_meta = []
    for target_name, vector in vectors:
        candidate = vector.float().clone()
        for existing in basis_vectors:
            candidate = candidate - torch.dot(candidate, existing) * existing
        norm = candidate.norm()
        if norm <= 1e-6:
            continue
        unit = candidate / norm
        strength = float(target_strengths.get(target_name, projection_strength))
        basis_vectors.append(unit)
        basis_meta.append(
            {
                "target": target_name,
                "strength": strength,
                "pre_orthogonalized_norm": float(vector.float().norm().item()),
            }
        )

    if not basis_vectors:
        raise ValueError("Could not build a non-empty basis from the supplied probe directions.")

    projector = eye.clone()
    for unit, meta in zip(basis_vectors, basis_meta):
        projector = projector - float(meta["strength"]) * torch.outer(unit, unit)
    basis = torch.stack(basis_vectors, dim=1)
    return projector, basis, basis_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-target-config", default="phase2/internal_eval_targets.json")
    parser.add_argument("--forget-csv", default="data/phase2/splits/forget.csv")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--out-dir", default="data/phase2/checkpoints")
    parser.add_argument("--run-name", default="probe_nullspace_joint_l5_l9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layers",
        default="",
        help="Optional global projection layer override, e.g. 4-10. Defaults to config layers.",
    )
    parser.add_argument(
        "--target-layers",
        default="",
        help=(
            "Optional per-target layer override, e.g. "
            "host_tropism=5-9,coronaviridae=4-10. Overrides --layers for named targets."
        ),
    )
    parser.add_argument(
        "--projection-strength",
        type=float,
        default=1.0,
        help="Soft projection strength lambda in I - lambda * BB^T. 1.0 is the current hard null-space.",
    )
    parser.add_argument(
        "--target-strengths",
        default="",
        help="Optional per-target strengths, e.g. host_tropism=1.0,coronaviridae=1.25.",
    )
    parser.add_argument(
        "--module-scope",
        choices=sorted(MODULE_SCOPE_SUFFIXES),
        default="all",
        help="Which residual-writer module family to project.",
    )
    parser.add_argument(
        "--module-suffixes",
        default="",
        help="Comma-separated module suffix allowlist. Overrides --module-scope when set.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Unused; accepted for runner compatibility.")
    parser.add_argument("--max-length", type=int, default=512, help="Unused; accepted for runner compatibility.")
    parser.add_argument("--save-steps", default="", help="Unused; accepted for runner compatibility.")
    args = parser.parse_args()

    t0 = time.time()
    target_specs = load_target_specs(args.internal_target_config)
    target_specs = apply_layer_overrides(target_specs, args.layers, args.target_layers)
    layers = sorted({layer for spec in target_specs for layer in spec["layers"]})
    target_names = [spec["name"] for spec in target_specs]
    target_strengths = parse_target_strengths(args.target_strengths)
    suffixes = (
        tuple(part.strip() for part in args.module_suffixes.split(",") if part.strip())
        if args.module_suffixes
        else MODULE_SCOPE_SUFFIXES[args.module_scope]
    )
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    model.eval()

    layer_bases: Dict[int, torch.Tensor] = {}
    projection_ranks: Dict[str, int] = {}
    layer_target_paths: Dict[str, Dict[str, str]] = {}
    for layer in layers:
        named_vectors = []
        target_paths = {}
        for spec in target_specs:
            if layer not in spec["layers"]:
                continue
            probe = load_probe(spec["probe_dir"], layer, device=args.device)
            named_vectors.append((spec["name"], normalized_raw_probe_direction(probe)))
            target_paths[spec["name"]] = probe["path"]
        basis = orthonormal_basis([vector for _name, vector in named_vectors])
        layer_bases[layer] = basis
        projection_ranks[str(layer)] = int(basis.shape[1])
        layer_target_paths[str(layer)] = target_paths
        print(
            f"[probe-nullspace] layer={layer} targets={sorted(target_paths)} "
            f"basis_rank={basis.shape[1]}"
        )

    modules_by_layer = discover_projection_modules(model, layers, suffixes)
    saved_tensors = {}
    projection_modules: Dict[str, List[str]] = {}
    layer_basis_meta: Dict[str, List[dict]] = {}
    with torch.no_grad():
        for layer in layers:
            named_vectors = []
            for spec in target_specs:
                if layer not in spec["layers"]:
                    continue
                probe = load_probe(spec["probe_dir"], layer, device=args.device)
                named_vectors.append((spec["name"], normalized_raw_probe_direction(probe)))
            projector, basis, basis_meta = soft_projection_matrix(
                named_vectors,
                projection_strength=args.projection_strength,
                target_strengths=target_strengths,
                hidden_dim=int(model.blocks[0].pre_norm.scale.shape[0]),
                device=args.device,
            )
            layer_bases[layer] = basis
            projection_ranks[str(layer)] = int(basis.shape[1])
            layer_basis_meta[str(layer)] = basis_meta
            selected_modules = modules_by_layer[layer]
            if not selected_modules:
                raise RuntimeError(f"No residual-writer modules matched the allowlist for layer {layer}.")
            projection_modules[str(layer)] = [item["full_name"] for item in selected_modules]
            for item in selected_modules:
                module = item["module"]
                module.weight.copy_((projector @ module.weight.float()).to(module.weight.dtype))
                saved_tensors[f"{item['full_name']}.weight"] = module.weight.detach().cpu()
                if module.bias is not None:
                    module.bias.copy_((projector @ module.bias.float()).to(module.bias.dtype))
                    saved_tensors[f"{item['full_name']}.bias"] = module.bias.detach().cpu()
                print(f"[probe-nullspace] projected {item['full_name']}")

    ckpt_path = os.path.join(run_dir, "weights.safetensors")
    save_file(saved_tensors, ckpt_path)
    meta = {
        "method": "probe_nullspace",
        "condition": "localized",
        "layers": layers,
        "target_names": target_names,
        "internal_target_config": args.internal_target_config,
        "layer_override": args.layers,
        "target_layer_overrides": parse_target_map(args.target_layers),
        "projection_strength": args.projection_strength,
        "target_strengths": target_strengths,
        "module_scope": args.module_scope,
        "module_suffixes": list(suffixes),
        "projection_ranks": projection_ranks,
        "projection_modules": projection_modules,
        "layer_basis_meta": layer_basis_meta,
        "layer_probe_paths": layer_target_paths,
        "forget_csv": args.forget_csv,
        "retain_csv": args.retain_csv,
        "init_source": "base_model",
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(
            {
                "layers": layers,
                "target_names": target_names,
                "projection_strength": args.projection_strength,
                "target_strengths": target_strengths,
                "module_scope": args.module_scope,
                "module_suffixes": list(suffixes),
                "projection_ranks": projection_ranks,
                "projection_modules": projection_modules,
                "layer_basis_meta": layer_basis_meta,
            },
            f,
            indent=2,
        )
    print(f"[probe-nullspace] saved {len(saved_tensors)} tensors to {ckpt_path}")


if __name__ == "__main__":
    main()
