"""Shared helpers for Phase 2 probe-guided unlearning methods."""
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

DEFAULT_LOCALIZED_LAYERS = [5, 6, 7, 8, 9]


def get_localized_layers(path: str) -> List[int]:
    if not path or not os.path.exists(path):
        return list(DEFAULT_LOCALIZED_LAYERS)
    with open(path) as f:
        payload = json.load(f)
    return sorted(set(int(layer) for layer in payload.get("layers", DEFAULT_LOCALIZED_LAYERS)))


def parse_layers(spec: str) -> List[int]:
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_target_specs(
    config_path: str,
    default_localized_layers_path: str = "data/family_targets/coronaviridae/localized_layers.json",
) -> List[dict]:
    with open(config_path) as f:
        payload = json.load(f)

    targets = []
    for entry in payload.get("targets", []):
        layers = parse_layers(entry.get("layers", "5-9"))
        localized_layers_path = entry.get("localized_layers_path", default_localized_layers_path)
        targets.append(
            {
                "name": entry["name"],
                "manifest": entry["manifest"],
                "probe_dir": entry["probe_dir"],
                "layers": layers,
                "localized_layers_path": localized_layers_path,
                "localized_layers": [
                    layer for layer in get_localized_layers(localized_layers_path) if layer in layers
                ],
            }
        )
    if not targets:
        raise ValueError(f"No targets found in {config_path}")
    return targets


def load_probe(probe_dir: str, layer_idx: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    path = Path(probe_dir) / f"layer_{layer_idx}.npz"
    data = np.load(path)
    return {
        "coef": torch.from_numpy(data["coef"].astype(np.float32)).to(device),
        "intercept": torch.from_numpy(data["intercept"].astype(np.float32)).to(device),
        "mean": torch.from_numpy(data["scaler_mean"].astype(np.float32)).to(device),
        "scale": torch.from_numpy(data["scaler_scale"].astype(np.float32)).to(device),
        "path": str(path),
    }


def normalized_standard_probe_direction(probe: Dict[str, torch.Tensor]) -> torch.Tensor:
    coef = probe["coef"].reshape(-1).float()
    return coef / coef.norm().clamp(min=1e-8)


def normalized_raw_probe_direction(probe: Dict[str, torch.Tensor]) -> torch.Tensor:
    coef = probe["coef"].reshape(-1).float()
    scale = probe["scale"].reshape(-1).float().clamp(min=1e-12)
    direction = coef / scale
    return direction / direction.norm().clamp(min=1e-8)


def orthonormal_basis(vectors: List[torch.Tensor], tol: float = 1e-6) -> torch.Tensor:
    basis: List[torch.Tensor] = []
    for vector in vectors:
        candidate = vector.float().clone()
        for existing in basis:
            candidate = candidate - torch.dot(candidate, existing) * existing
        norm = candidate.norm()
        if norm <= tol:
            continue
        basis.append(candidate / norm)
    if not basis:
        raise ValueError("Could not build a non-empty basis from the supplied probe directions.")
    return torch.stack(basis, dim=1)


def projection_matrix(basis: torch.Tensor) -> torch.Tensor:
    hidden_dim = basis.shape[0]
    eye = torch.eye(hidden_dim, device=basis.device, dtype=basis.dtype)
    return eye - basis @ basis.T


def apply_checkpoint(model, ckpt_path: str, checkpoint_format: str = "auto") -> None:
    from phase2.checkpoint_io import apply_checkpoint as apply_phase2_checkpoint

    apply_phase2_checkpoint(
        model,
        ckpt_path,
        checkpoint_format=checkpoint_format,
        log_prefix="probe",
    )


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
