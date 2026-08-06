"""Shared LoRA and supervised-task helpers for benchmark evaluation."""
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


class LoRALinear(nn.Module):
    """Wrap a frozen Linear layer with trainable low-rank adapters."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.zeros(rank, linear.in_features, device=linear.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=linear.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        adapter_in = self.dropout(x).float()
        adapter_out = (adapter_in @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + adapter_out.to(base.dtype)


@dataclass
class LabelEncoding:
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _replace_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_lora_all_blocks(
    model,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
) -> Tuple[List[nn.Parameter], List[str]]:
    """Inject LoRA into every Linear under every Evo transformer block."""
    freeze_all(model)
    adapter_params: List[nn.Parameter] = []
    module_names: List[str] = []

    for block_idx, block in enumerate(model.blocks):
        for module_path, module in list(block.named_modules()):
            if not module_path or not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
                continue
            parts = module_path.split(".")
            parent = block
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
            child_name = parts[-1]
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            _replace_child(parent, child_name, lora_layer)
            adapter_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            module_names.append(f"blocks.{block_idx}.{module_path}")

    for param in adapter_params:
        param.requires_grad_(True)
    return adapter_params, module_names


def remove_lora_adapters(model) -> None:
    """Restore wrapped Linear modules in-place after a task finishes."""
    for block in model.blocks:
        for module_path, module in list(block.named_modules()):
            if not module_path or not isinstance(module, LoRALinear):
                continue
            parts = module_path.split(".")
            parent = block
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
            _replace_child(parent, parts[-1], module.linear)


def merge_lora_adapters(model) -> list[str]:
    """Merge LoRA updates into base Linear weights and remove wrappers in-place."""
    merged: list[str] = []
    for block_idx, block in enumerate(model.blocks):
        for module_path, module in list(block.named_modules()):
            if not module_path or not isinstance(module, LoRALinear):
                continue
            parts = module_path.split(".")
            parent = block
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
            child_name = parts[-1]
            merged_linear = module.linear
            delta = (module.lora_B.float() @ module.lora_A.float()) * module.scale
            merged_linear.weight.data.add_(delta.to(merged_linear.weight.dtype).to(merged_linear.weight.device))
            _replace_child(parent, child_name, merged_linear)
            merged.append(f"blocks.{block_idx}.{module_path}")
    return merged


def count_trainable(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def count_total(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def encode_labels(labels: Iterable[str]) -> Tuple[np.ndarray, LabelEncoding]:
    ordered = sorted(set(str(label) for label in labels))
    encoding = LabelEncoding(
        label_to_id={label: idx for idx, label in enumerate(ordered)},
        id_to_label={idx: label for idx, label in enumerate(ordered)},
    )
    encoded = np.array([encoding.label_to_id[str(label)] for label in labels], dtype=np.int64)
    return encoded, encoding


class PooledEvoClassifier(nn.Module):
    """Mean-pool final normalized Evo states and apply a supervised task head."""

    def __init__(self, base_model, hidden_dim: int, output_dim: int, problem_type: str):
        super().__init__()
        self.base_model = base_model
        self.problem_type = problem_type
        self.head = nn.Linear(hidden_dim, output_dim)
        self._captured = None
        last_idx = len(base_model.blocks) - 1

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured = self.base_model.norm(hidden)

        self._hook = base_model.blocks[last_idx].register_forward_hook(capture)

    def close(self) -> None:
        self._hook.remove()

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._captured = None
        _logits, _ = self.base_model(input_ids, padding_mask=mask)
        if self._captured is None:
            raise RuntimeError("Failed to capture final Evo hidden states")
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (self._captured * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(pooled.float())


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    num_classes: int,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": None,
        "auprc": None,
    }
    observed = np.unique(y_true)
    if y_score is None or len(observed) < 2:
        return metrics
    try:
        if num_classes == 2:
            positive = y_score[:, 1] if y_score.ndim == 2 else y_score
            metrics["auroc"] = float(roc_auc_score(y_true, positive))
            metrics["auprc"] = float(average_precision_score(y_true, positive))
        elif set(observed.tolist()) == set(range(num_classes)):
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            metrics["auroc"] = float(
                roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
            )
            metrics["auprc"] = float(average_precision_score(y_bin, y_score, average="macro"))
    except ValueError:
        pass
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    result: Dict[str, Optional[float]] = {
        "mse": mse,
        "rmse": rmse,
        "r2": None,
        "pearson": None,
    }
    if len(y_true) >= 2:
        try:
            result["r2"] = float(r2_score(y_true, y_pred))
        except ValueError:
            pass
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            result["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    return result
