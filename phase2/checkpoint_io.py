"""Checkpoint save/load helpers for Phase 2 unlearning.

The helpers here keep the old partial-safetensors behavior working while adding
delta checkpoints, disk guards, and atomic writes for new probe-guided runs.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_TRAINABLE_MODULE_SUFFIXES = [
    "inner_mha_cls.out_proj",
    "out_filter_dense",
    "mlp.l3",
]


@dataclass
class SaveResult:
    path: str
    saved: bool
    skipped_reason: str | None
    free_disk_gb: float
    min_free_disk_gb: float
    tensor_count: int
    tensor_names: list[str]
    checkpoint_policy: str


@dataclass
class ApplyResult:
    path: str
    checkpoint_policy: str
    tensor_mode: str
    applied_count: int
    missing_keys: list[str]


def parse_module_suffixes(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return list(DEFAULT_TRAINABLE_MODULE_SUFFIXES)
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in spec if str(part).strip()]
    return parts or list(DEFAULT_TRAINABLE_MODULE_SUFFIXES)


def is_all_suffixes(suffixes: Sequence[str]) -> bool:
    return any(suffix.lower() == "all" for suffix in suffixes)


def tensor_matches_layer_and_suffix(key: str, layers: Iterable[int], suffixes: Sequence[str]) -> bool:
    layer_set = set(int(layer) for layer in layers)
    for layer_idx in layer_set:
        prefix = f"blocks.{layer_idx}."
        if not key.startswith(prefix):
            continue
        if is_all_suffixes(suffixes):
            return True
        parameter_name = key[len(prefix) :]
        module_name = parameter_name.rsplit(".", 1)[0]
        return any(module_name == suffix or module_name.endswith(f".{suffix}") for suffix in suffixes)
    return False


def select_tensor_names(
    model: torch.nn.Module,
    layers: Iterable[int] | None,
    suffixes: Sequence[str] | None,
    *,
    adapter_only: bool = False,
) -> list[str]:
    suffix_list = parse_module_suffixes(suffixes)
    names = []
    for name, param in model.named_parameters():
        if adapter_only:
            lowered = name.lower()
            if not any(marker in lowered for marker in ("adapter", "lora", "ia3")):
                continue
        elif layers is not None and not tensor_matches_layer_and_suffix(name, layers, suffix_list):
            continue
        names.append(name)
    return sorted(names)


def set_trainable_by_suffixes(
    model: torch.nn.Module,
    layers: Iterable[int],
    suffixes: Sequence[str] | str | None,
) -> list[str]:
    suffix_list = parse_module_suffixes(suffixes)
    selected = set(select_tensor_names(model, layers, suffix_list))
    for name, param in model.named_parameters():
        param.requires_grad_(name in selected)
    if not selected:
        raise ValueError(
            f"No trainable tensors matched layers={list(layers)} suffixes={','.join(suffix_list)}"
        )
    return sorted(selected)


def snapshot_state(
    model: torch.nn.Module,
    tensor_names: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    selected = set(tensor_names) if tensor_names is not None else None
    result = {}
    for key, value in model.state_dict().items():
        if selected is not None and key not in selected:
            continue
        result[key] = value.detach().cpu().clone()
    return result


def free_disk_gb_for_path(path: str | os.PathLike[str]) -> float:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    return usage.free / (1024**3)


def _metadata_as_strings(metadata: Mapping[str, object] | None) -> dict[str, str]:
    result = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        result[str(key)] = str(value)
    return result


def read_safetensors_metadata(path: str | os.PathLike[str]) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def infer_checkpoint_policy(path: str, requested: str = "auto") -> tuple[str, str]:
    if requested != "auto":
        if requested == "delta":
            return requested, "delta"
        return requested, "absolute"
    metadata = read_safetensors_metadata(path)
    policy = metadata.get("checkpoint_policy") or metadata.get("save_policy") or "selected_modules"
    tensor_mode = metadata.get("tensor_mode")
    if tensor_mode == "delta" or policy == "delta":
        return "delta", "delta"
    if policy == "standalone_lora_reverse":
        return policy, "custom"
    if policy == "adapter":
        return "adapter", "absolute"
    if policy == "full":
        return "full", "absolute"
    return "selected_modules", "absolute"


def atomic_save_safetensors(
    tensors: Mapping[str, torch.Tensor],
    out_path: str,
    *,
    metadata: Mapping[str, object] | None = None,
    min_free_disk_gb: float = 0.0,
) -> SaveResult:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    free_gb = free_disk_gb_for_path(out_path)
    checkpoint_policy = str((metadata or {}).get("checkpoint_policy", "unknown"))
    tensor_names = sorted(tensors.keys())
    if min_free_disk_gb > 0 and free_gb < min_free_disk_gb:
        return SaveResult(
            path=out_path,
            saved=False,
            skipped_reason="low_disk",
            free_disk_gb=free_gb,
            min_free_disk_gb=min_free_disk_gb,
            tensor_count=len(tensor_names),
            tensor_names=tensor_names,
            checkpoint_policy=checkpoint_policy,
        )
    tmp_path = f"{out_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        save_file(dict(tensors), tmp_path, metadata=_metadata_as_strings(metadata))
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return SaveResult(
        path=out_path,
        saved=True,
        skipped_reason=None,
        free_disk_gb=free_gb,
        min_free_disk_gb=min_free_disk_gb,
        tensor_count=len(tensor_names),
        tensor_names=tensor_names,
        checkpoint_policy=checkpoint_policy,
    )


def collect_checkpoint_tensors(
    model: torch.nn.Module,
    *,
    policy: str,
    layers: Iterable[int] | None = None,
    suffixes: Sequence[str] | str | None = None,
    init_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    suffix_list = parse_module_suffixes(suffixes)
    if policy == "full":
        names = sorted(model.state_dict().keys())
    elif policy in {"selected_modules", "delta"}:
        if layers is None:
            raise ValueError(f"layers are required for save policy {policy}")
        names = select_tensor_names(model, layers, suffix_list)
    elif policy == "adapter":
        names = select_tensor_names(model, layers, suffix_list, adapter_only=True)
        if not names:
            raise ValueError("save-policy=adapter requested, but no adapter/LoRA/IA3 tensors were found")
    else:
        raise ValueError(f"Unknown save policy: {policy}")

    state = model.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for key in names:
        value = state[key].detach().cpu()
        if policy == "delta":
            if init_state is None or key not in init_state:
                raise ValueError(f"Missing initialization tensor for delta checkpoint key: {key}")
            tensors[key] = (value.float() - init_state[key].float()).cpu()
        else:
            tensors[key] = value.to(torch.bfloat16).cpu() if value.is_floating_point() else value.cpu()
    if not tensors:
        raise ValueError(f"No tensors selected for save policy {policy}")
    return tensors


def save_checkpoint(
    model: torch.nn.Module,
    out_path: str,
    *,
    policy: str,
    layers: Iterable[int] | None = None,
    suffixes: Sequence[str] | str | None = None,
    init_state: Mapping[str, torch.Tensor] | None = None,
    min_free_disk_gb: float = 0.0,
    metadata: Mapping[str, object] | None = None,
) -> SaveResult:
    tensors = collect_checkpoint_tensors(
        model,
        policy=policy,
        layers=layers,
        suffixes=suffixes,
        init_state=init_state,
    )
    merged_metadata = {
        **(metadata or {}),
        "checkpoint_policy": policy,
        "tensor_mode": "delta" if policy == "delta" else "absolute",
    }
    return atomic_save_safetensors(
        tensors,
        out_path,
        metadata=merged_metadata,
        min_free_disk_gb=min_free_disk_gb,
    )


def _load_compact_lora_payload(ckpt_path: str) -> tuple[dict[str, str], dict[str, object]]:
    metadata = read_safetensors_metadata(ckpt_path)
    provenance_path = Path(ckpt_path).with_name("provenance.json")
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing compact checkpoint provenance file: {provenance_path}")
    payload = json.loads(provenance_path.read_text())
    return metadata, payload


def _module_name_from_adapter_key(key: str) -> str:
    module = key[len("base_model.") :] if key.startswith("base_model.") else key
    if module.endswith(".lora_A"):
        return module[: -len(".lora_A")]
    if module.endswith(".lora_B"):
        return module[: -len(".lora_B")]
    raise ValueError(f"Unsupported adapter key: {key}")


def _load_adapter_updates(adapter_path: str, scale: float) -> dict[str, torch.Tensor]:
    payload = torch.load(adapter_path, map_location="cpu")
    state = payload["state_dict"]
    modules = sorted({_module_name_from_adapter_key(key) for key in state if key.endswith(".lora_A")})
    updates: dict[str, torch.Tensor] = {}
    for module in modules:
        A = state[f"base_model.{module}.lora_A"].float()
        B = state[f"base_model.{module}.lora_B"].float()
        updates[module] = (B @ A) * scale
    return updates


def _build_random_orientation(delta: torch.Tensor, *, seed: int) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(delta.float())
    rank = int((singular_values > 0).sum().item())
    if rank == 0:
        return torch.zeros_like(delta)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.randn(delta.shape[0], rank, generator=generator)
    right = torch.randn(delta.shape[1], rank, generator=generator)
    q_left, _ = torch.linalg.qr(left, mode="reduced")
    q_right, _ = torch.linalg.qr(right, mode="reduced")
    return (q_left[:, :rank] * singular_values[:rank].unsqueeze(0)) @ q_right[:, :rank].T


def _build_random_orientation_from_singular_values(
    *,
    out_dim: int,
    in_dim: int,
    singular_values: Sequence[float],
    seed: int,
) -> torch.Tensor:
    rank = len([value for value in singular_values if float(value) > 0])
    if rank == 0:
        return torch.zeros((out_dim, in_dim), dtype=torch.float32)
    s = torch.tensor([float(value) for value in singular_values[:rank]], dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.randn(out_dim, rank, generator=generator)
    right = torch.randn(in_dim, rank, generator=generator)
    q_left, _ = torch.linalg.qr(left, mode="reduced")
    q_right, _ = torch.linalg.qr(right, mode="reduced")
    return (q_left[:, :rank] * s.unsqueeze(0)) @ q_right[:, :rank].T


def _apply_compact_reverse_lora_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    log_prefix: str,
) -> ApplyResult:
    metadata, payload = _load_compact_lora_payload(ckpt_path)
    adapter_path = str(payload["source_adapter_path"])
    eta = float(payload["eta"])
    scale = float(payload.get("lora_scale", 1.0))
    mappings = list(payload.get("mappings") or [])
    state_dict = model.state_dict()
    applied = 0
    missing: list[str] = []
    updates: dict[str, torch.Tensor] = {}
    full_updates: dict[str, torch.Tensor] | None = None
    for idx, row in enumerate(mappings):
        source_module = str(row["source_module"])
        target_module = str(row["target_module"])
        policy = str(row.get("policy", "reverse_source_direction"))
        weight_key = f"{target_module}.weight"
        if weight_key not in state_dict:
            missing.append(weight_key)
            continue
        if policy == "random_orientation_same_slot" and row.get("singular_values"):
            delta = _build_random_orientation_from_singular_values(
                out_dim=state_dict[weight_key].shape[0],
                in_dim=state_dict[weight_key].shape[1],
                singular_values=row["singular_values"],
                seed=int(row["orientation_seed"]),
            )
        else:
            if source_module not in updates:
                if full_updates is None:
                    full_updates = _load_adapter_updates(adapter_path, scale)
                updates[source_module] = full_updates[source_module]
            base_delta = updates[source_module]
            if policy == "random_orientation_same_slot":
                seed = int(row["orientation_seed"])
                delta = _build_random_orientation(base_delta, seed=seed)
            else:
                delta = base_delta
        state_dict[weight_key].add_((-eta * delta).to(state_dict[weight_key].dtype).to(state_dict[weight_key].device))
        applied += 1
    print(
        f"[{log_prefix}] applied {applied} compact reverse-LoRA module deltas from {ckpt_path} "
        f"policy={metadata.get('checkpoint_policy','')} source={adapter_path}"
    )
    return ApplyResult(
        path=ckpt_path,
        checkpoint_policy="standalone_lora_reverse",
        tensor_mode="custom",
        applied_count=applied,
        missing_keys=missing,
    )


def apply_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    checkpoint_format: str = "auto",
    strict_adapter: bool = True,
    log_prefix: str = "checkpoint",
) -> ApplyResult:
    policy, tensor_mode = infer_checkpoint_policy(ckpt_path, checkpoint_format)
    if policy == "standalone_lora_reverse":
        return _apply_compact_reverse_lora_checkpoint(model, ckpt_path, log_prefix=log_prefix)
    tensors = load_file(ckpt_path)
    state_dict = model.state_dict()
    missing = []
    applied = 0
    for key, value in tensors.items():
        if key not in state_dict:
            missing.append(key)
            continue
        target = state_dict[key]
        value = value.to(target.dtype).to(target.device)
        if tensor_mode == "delta":
            target.add_(value)
        else:
            target.copy_(value)
        applied += 1
    if missing and (policy == "adapter" and strict_adapter):
        raise KeyError(f"{len(missing)} adapter checkpoint tensors not present in model: {missing[:5]}")
    if missing:
        print(f"[{log_prefix}] skipped {len(missing)} checkpoint tensors not present in model")
    print(f"[{log_prefix}] applied {applied} tensors from {ckpt_path} policy={policy} mode={tensor_mode}")
    return ApplyResult(
        path=ckpt_path,
        checkpoint_policy=policy,
        tensor_mode=tensor_mode,
        applied_count=applied,
        missing_keys=missing,
    )
