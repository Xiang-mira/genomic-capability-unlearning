"""Smoke tests for Phase 2 checkpoint I/O and run metadata."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2 import checkpoint_io
from phase2.checkpoint_io import (
    apply_checkpoint,
    atomic_save_safetensors,
    save_checkpoint,
    set_trainable_by_suffixes,
    snapshot_state,
)
from phase2.run_metadata import build_run_metadata, write_metadata


class TinySubmodule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = torch.nn.Linear(4, 4)


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l3 = torch.nn.Linear(4, 4)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner_mha_cls = TinySubmodule()
        self.out_filter_dense = torch.nn.Linear(4, 4)
        self.mlp = TinyMLP()
        self.other = torch.nn.Linear(4, 4)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
        self.adapter = torch.nn.Linear(4, 4)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_state_close(model: torch.nn.Module, expected: dict[str, torch.Tensor], keys: list[str]) -> None:
    state = model.state_dict()
    for key in keys:
        if not torch.allclose(state[key].float(), expected[key].float(), atol=5e-3, rtol=5e-3):
            raise AssertionError(f"state mismatch for {key}")


def perturb(model: torch.nn.Module, names: list[str], amount: float) -> None:
    with torch.no_grad():
        state = model.state_dict()
        for name in names:
            state[name].add_(amount)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/phase2/audits/task0_3_20260713/smoke_checkpoint_io")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    suffixes = "inner_mha_cls.out_proj,out_filter_dense,mlp.l3"
    model = TinyModel()
    selected_names = set_trainable_by_suffixes(model, [0], suffixes)
    init_state = snapshot_state(model, selected_names)
    base_state = clone_state(model)

    perturb(model, selected_names, 0.125)
    expected_after_perturb = clone_state(model)

    selected_path = out_dir / "selected_modules.safetensors"
    selected_result = save_checkpoint(
        model,
        str(selected_path),
        policy="selected_modules",
        layers=[0],
        suffixes=suffixes,
        min_free_disk_gb=0,
        metadata={"created_by": "smoke_checkpoint_io"},
    )
    restored = TinyModel()
    apply_checkpoint(restored, str(selected_path), checkpoint_format="auto", log_prefix="smoke")
    assert_state_close(restored, expected_after_perturb, selected_names)
    results["selected_modules"] = selected_result.__dict__

    delta_path = out_dir / "delta.safetensors"
    delta_result = save_checkpoint(
        model,
        str(delta_path),
        policy="delta",
        layers=[0],
        suffixes=suffixes,
        init_state=init_state,
        min_free_disk_gb=0,
        metadata={"created_by": "smoke_checkpoint_io"},
    )
    restored_delta = TinyModel()
    restored_delta.load_state_dict(base_state)
    apply_checkpoint(restored_delta, str(delta_path), checkpoint_format="auto", log_prefix="smoke")
    assert_state_close(restored_delta, expected_after_perturb, selected_names)
    results["delta"] = delta_result.__dict__

    full_path = out_dir / "full.safetensors"
    full_result = save_checkpoint(
        model,
        str(full_path),
        policy="full",
        min_free_disk_gb=0,
        metadata={"created_by": "smoke_checkpoint_io"},
    )
    restored_full = TinyModel()
    apply_checkpoint(restored_full, str(full_path), checkpoint_format="auto", log_prefix="smoke")
    assert_state_close(restored_full, expected_after_perturb, list(expected_after_perturb.keys()))
    results["full"] = full_result.__dict__

    adapter_path = out_dir / "adapter.safetensors"
    adapter_result = save_checkpoint(
        model,
        str(adapter_path),
        policy="adapter",
        min_free_disk_gb=0,
        metadata={"created_by": "smoke_checkpoint_io"},
    )
    restored_adapter = TinyModel()
    restored_adapter.load_state_dict(base_state)
    apply_checkpoint(restored_adapter, str(adapter_path), checkpoint_format="auto", log_prefix="smoke")
    assert_state_close(restored_adapter, expected_after_perturb, ["adapter.weight", "adapter.bias"])
    results["adapter"] = adapter_result.__dict__

    low_disk_path = out_dir / "low_disk_skip.safetensors"
    low_disk_result = save_checkpoint(
        model,
        str(low_disk_path),
        policy="selected_modules",
        layers=[0],
        suffixes=suffixes,
        min_free_disk_gb=10**9,
        metadata={"created_by": "smoke_checkpoint_io"},
    )
    if low_disk_path.exists():
        raise AssertionError("low disk save unexpectedly wrote a final file")
    results["low_disk_skip"] = low_disk_result.__dict__

    legacy_path = out_dir / "legacy_absolute.safetensors"
    save_file({"blocks.0.other.weight": model.state_dict()["blocks.0.other.weight"]}, str(legacy_path))
    legacy_restored = TinyModel()
    legacy_restored.load_state_dict(base_state)
    apply_checkpoint(legacy_restored, str(legacy_path), checkpoint_format="auto", log_prefix="smoke")
    assert_state_close(legacy_restored, expected_after_perturb, ["blocks.0.other.weight"])
    results["legacy_absolute"] = {"path": str(legacy_path), "passed": True}

    original_save_file = checkpoint_io.save_file

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated save failure")

    partial_path = out_dir / "partial_failure.safetensors"
    checkpoint_io.save_file = boom
    try:
        try:
            atomic_save_safetensors(
                {"x": torch.ones(1)},
                str(partial_path),
                metadata={"checkpoint_policy": "selected_modules"},
                min_free_disk_gb=0,
            )
            raise AssertionError("simulated save failure did not raise")
        except RuntimeError:
            leftovers = list(out_dir.glob("partial_failure.safetensors.tmp.*"))
            if leftovers or partial_path.exists():
                raise AssertionError("partial save left a final or temp file")
            results["partial_cleanup"] = {"passed": True}
    finally:
        checkpoint_io.save_file = original_save_file

    metadata = build_run_metadata(
        args={"smoke": True, "out_dir": str(out_dir)},
        source_checkpoint="tiny_base",
        init_checkpoint="tiny_base",
        output_checkpoint=str(delta_path),
        trainable_modules=suffixes.split(","),
        trainable_tensor_names=selected_names,
        trainable_param_count=sum(model.state_dict()[name].numel() for name in selected_names),
        loss_layers=[0],
        seed=42,
        save_policy="delta",
        checkpoint_policy="delta",
        extra={"smoke_results": results},
    )
    write_metadata(out_dir / "meta.json", metadata)
    (out_dir / "smoke_checkpoint_io.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[smoke-checkpoint-io] wrote {out_dir}")


if __name__ == "__main__":
    main()
