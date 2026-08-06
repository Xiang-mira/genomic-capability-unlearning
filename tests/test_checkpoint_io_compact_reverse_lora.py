from __future__ import annotations

import json
from pathlib import Path

import torch

from phase2.checkpoint_io import apply_checkpoint, atomic_save_safetensors


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projections = torch.nn.Linear(3, 2, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock()])


def test_apply_checkpoint_supports_compact_reverse_lora(tmp_path: Path) -> None:
    adapter_path = tmp_path / "source_adapter.pt"
    A = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    B = torch.tensor([[4.0], [5.0]], dtype=torch.float32)
    torch.save(
        {
            "state_dict": {
                "base_model.blocks.0.projections.lora_A": A,
                "base_model.blocks.0.projections.lora_B": B,
            }
        },
        adapter_path,
    )

    ckpt_dir = tmp_path / "compact_ckpt"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "weights.safetensors"
    atomic_save_safetensors(
        {"__compact_anchor__": torch.tensor([0], dtype=torch.uint8)},
        str(ckpt_path),
        metadata={
            "checkpoint_policy": "standalone_lora_reverse",
            "arm_id": "source_subspace_intervention_eta1",
            "arm_type": "source_subspace_intervention",
            "source_run_id": "test_source",
            "eta": 0.25,
        },
    )
    (ckpt_dir / "provenance.json").write_text(
        json.dumps(
            {
                "arm_id": "source_subspace_intervention_eta1",
                "arm_type": "source_subspace_intervention",
                "eta": 0.25,
                "source_run_id": "test_source",
                "source_adapter_path": str(adapter_path),
                "lora_scale": 2.0,
                "mappings": [
                    {
                        "source_module": "blocks.0.projections",
                        "target_module": "blocks.0.projections",
                        "policy": "reverse_source_direction",
                    }
                ],
            }
        )
    )

    model = TinyModel()
    before = model.blocks[0].projections.weight.detach().clone()

    result = apply_checkpoint(model, str(ckpt_path), log_prefix="compact-test")

    delta = (B @ A) * 2.0
    expected = before - 0.25 * delta
    assert torch.allclose(model.blocks[0].projections.weight.detach(), expected)
    assert result.checkpoint_policy == "standalone_lora_reverse"
    assert result.tensor_mode == "custom"
    assert result.applied_count == 1
