"""Checkpoint save/load helpers for Phase 2 unlearning.

The helpers here keep the old partial-safetensors behavior working while adding
delta checkpoints, disk guards, and atomic writes for new probe-guided runs.
"""
from __future__ import annotations

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


def apply_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    checkpoint_format: str = "auto",
    strict_adapter: bool = True,
    log_prefix: str = "checkpoint",
) -> ApplyResult:
    policy, tensor_mode = infer_checkpoint_policy(ckpt_path, checkpoint_format)
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
