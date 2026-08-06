"""Minimal training-mode smoke tests for Experiment 3 relearning launch."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint
from phase2.eval_benchmarks import (
    compute_loss_and_outputs,
    deterministic_stratified_subset,
    evaluate_model,
    labels_for_split,
    load_trainable_state_dict,
    read_benchmark_manifest,
    save_best_checkpoint,
    set_seed,
    split_records,
)
from phase2.lora_utils import PooledEvoClassifier, count_total, count_trainable, encode_labels, inject_lora_all_blocks
from phase2.utils import tokenize_batch


TASK = "hvue_human_host_tropism"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tensor_changed(before: torch.Tensor, after: torch.Tensor) -> bool:
    return bool((before.float().cpu() - after.detach().float().cpu()).abs().max().item() > 0.0)


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def count_lora_modules(model: torch.nn.Module) -> int:
    from phase2.lora_utils import LoRALinear

    return sum(1 for module in model.modules() if isinstance(module, LoRALinear))


def first_named_param(module: torch.nn.Module, predicate) -> tuple[str, torch.nn.Parameter]:
    for name, param in module.named_parameters():
        if predicate(name, param):
            return name, param
    raise RuntimeError("No parameter matched predicate")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    set_seed(args.seed)
    result: dict[str, Any] = {
        "mode": args.mode,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": " ".join(sys.argv),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_ckpt = output_path.parent / f"{args.mode}_temporary_checkpoint.pt"
    try:
        records = read_benchmark_manifest(
            args.benchmark_manifest,
            benchmark_scope="task",
            task_filter={TASK},
            requested_split_type="cluster_disjoint",
        )
        splits = split_records(records)
        train_records = deterministic_stratified_subset(splits["train"], args.max_train_rows, args.seed, TASK)
        val_records = deterministic_stratified_subset(splits["val"], args.max_val_rows, args.seed, f"{TASK}:smoke_val")
        _, label_encoding = encode_labels(record.label for record in records)
        train_labels = labels_for_split(train_records, label_encoding.label_to_id, "classification")
        val_labels = labels_for_split(val_records, label_encoding.label_to_id, "classification")

        model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
        model_load_key = "fresh_lora_model_load" if args.mode == "fresh_lora" else "full_ft_model_load"
        result[model_load_key] = "pass"
        lora_modules: list[str] = []
        if args.mode == "fresh_lora":
            _adapter_params, lora_modules = inject_lora_all_blocks(
                model,
                rank=args.lora_rank,
                alpha=args.lora_rank * 2,
                dropout=0.0,
            )
        else:
            for param in model.parameters():
                param.requires_grad_(True)

        hidden_dim = int(model.blocks[0].pre_norm.scale.shape[0])
        task_model = PooledEvoClassifier(model, hidden_dim, label_encoding.num_classes, "classification").to(args.device)
        for param in task_model.head.parameters():
            param.requires_grad_(True)

        trainable_named = [(name, param) for name, param in task_model.named_parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW([param for _name, param in trainable_named], lr=args.lr, weight_decay=0.0)
        tokenizer = CharLevelTokenizer(args.max_length)

        if args.mode == "fresh_lora":
            result["fresh_lora_parameters_trainable"] = "pass" if any(".lora_" in name for name, _ in trainable_named) else "fail"
            result["fresh_classification_head_trainable"] = "pass" if all(param.requires_grad for param in task_model.head.parameters()) else "fail"
            backbone_trainable = [
                name for name, param in task_model.named_parameters()
                if name.startswith("base_model.") and ".lora_" not in name and param.requires_grad
            ]
            result["base_backbone_frozen"] = "pass" if not backbone_trainable else "fail"
            result["optimizer_contains_only_lora_and_head"] = (
                "pass" if all(".lora_" in name or name.startswith("head.") for name, _ in trainable_named) else "fail"
            )
            lora_name, lora_param = first_named_param(task_model, lambda name, param: ".lora_B" in name and param.requires_grad)
            backbone_name, backbone_param = first_named_param(
                task_model,
                lambda name, param: name.startswith("base_model.") and ".lora_" not in name and param.is_floating_point(),
            )
        else:
            result["full_ft_backbone_trainable"] = (
                "pass" if any(name.startswith("base_model.") and param.requires_grad for name, param in task_model.named_parameters()) else "fail"
            )
            result["fresh_classification_head_trainable"] = "pass" if all(param.requires_grad for param in task_model.head.parameters()) else "fail"
            result["active_lora_adapter"] = "none" if count_lora_modules(task_model) == 0 else "present"
            result["optimizer_contains_backbone_parameters"] = (
                "pass" if any(name.startswith("base_model.") for name, _ in trainable_named) else "fail"
            )
            result["optimizer_contains_head_parameters"] = (
                "pass" if any(name.startswith("head.") for name, _ in trainable_named) else "fail"
            )
            backbone_name, backbone_param = first_named_param(
                task_model,
                lambda name, param: name.startswith("base_model.") and param.requires_grad and param.is_floating_point(),
            )
            lora_name, lora_param = backbone_name, backbone_param

        head_name, head_param = first_named_param(task_model, lambda name, param: name.startswith("head.") and param.requires_grad)
        lora_or_backbone_before = lora_param.detach().cpu().clone()
        head_before = head_param.detach().cpu().clone()
        backbone_before = backbone_param.detach().cpu().clone()

        task_model.train()
        finite_loss = True
        finite_logits = True
        for step in range(args.max_steps):
            batch = train_records[step : step + 1]
            target_np = train_labels[step : step + 1]
            ids, mask = tokenize_batch([record.sequence for record in batch], tokenizer, args.max_length, args.device)
            targets = torch.tensor(target_np, dtype=torch.long, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, logits = compute_loss_and_outputs(task_model, ids, mask, targets, "classification")
            finite_loss = finite_loss and finite_tensor(loss)
            finite_logits = finite_logits and finite_tensor(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for _name, param in trainable_named], 1.0)
            optimizer.step()
            del ids, mask, targets, loss, logits

        result["finite_loss"] = "pass" if finite_loss else "fail"
        result["finite_logits"] = "pass" if finite_logits else "fail"
        if args.mode == "fresh_lora":
            result["lora_parameter_changed_after_step"] = "pass" if tensor_changed(lora_or_backbone_before, lora_param) else "fail"
            result["frozen_backbone_unchanged"] = "pass" if not tensor_changed(backbone_before, backbone_param) else "fail"
        else:
            result["backbone_parameter_changed_after_step"] = "pass" if tensor_changed(backbone_before, backbone_param) else "fail"
        result["head_parameter_changed_after_step"] = "pass" if tensor_changed(head_before, head_param) else "fail"

        if args.mode == "full_ft":
            trainable_count = count_trainable(task_model)
            total_count = count_total(task_model)
            available_bytes = shutil.disk_usage(output_path.parent).free
            estimated_min_bytes = trainable_count * 2 + 5 * 1024**3
            result["trainable_parameter_count"] = trainable_count
            result["total_parameter_count"] = total_count
            result["trainable_parameter_fraction"] = trainable_count / max(total_count, 1)
            result["available_disk_space_bytes"] = available_bytes
            result["checkpoint_disk_space_check"] = "pass" if available_bytes > estimated_min_bytes else "fail"
            if result["checkpoint_disk_space_check"] != "pass":
                raise RuntimeError(
                    f"insufficient disk for temporary full-FT checkpoint: available={available_bytes} required_estimate={estimated_min_bytes}"
                )

        val_loss, val_metrics, val_predictions = evaluate_model(
            task_model,
            val_records,
            val_labels,
            tokenizer,
            1,
            args.max_length,
            args.device,
            "classification",
            label_encoding.num_classes,
            return_predictions=True,
        )
        before_reload_scores = val_predictions["y_score"].copy()
        save_best_checkpoint(str(temp_ckpt), task_model, {"step": args.max_steps, "val_loss": val_loss, "val_metrics": val_metrics})
        result["temporary_checkpoint_save"] = "pass" if temp_ckpt.exists() else "fail"
        result["temporary_checkpoint_size_bytes"] = temp_ckpt.stat().st_size if temp_ckpt.exists() else 0
        payload = torch.load(temp_ckpt, map_location="cpu")
        load_trainable_state_dict(task_model, payload["state_dict"], args.device)
        result["temporary_checkpoint_reload"] = "pass"
        _val_loss_2, _val_metrics_2, val_predictions_2 = evaluate_model(
            task_model,
            val_records,
            val_labels,
            tokenizer,
            1,
            args.max_length,
            args.device,
            "classification",
            label_encoding.num_classes,
            return_predictions=True,
        )
        result["reloaded_prediction_consistency"] = (
            "pass" if np.allclose(before_reload_scores, val_predictions_2["y_score"], rtol=0.0, atol=0.0) else "fail"
        )
        result["trainable_tensor_example"] = lora_name
        result["head_tensor_example"] = head_name
        result["backbone_tensor_example"] = backbone_name
        result["lora_module_count"] = len(lora_modules)
        result["elapsed_sec"] = time.time() - started
        required = [
            value for key, value in result.items()
            if key.endswith("_pass") or key in {
                "fresh_lora_model_load",
                "fresh_lora_parameters_trainable",
                "fresh_classification_head_trainable",
                "base_backbone_frozen",
                "optimizer_contains_only_lora_and_head",
                "lora_parameter_changed_after_step",
                "head_parameter_changed_after_step",
                "frozen_backbone_unchanged",
                "temporary_checkpoint_save",
                "temporary_checkpoint_reload",
                "reloaded_prediction_consistency",
                "finite_loss",
                "finite_logits",
                "full_ft_model_load",
                "full_ft_backbone_trainable",
                "active_lora_adapter",
                "optimizer_contains_backbone_parameters",
                "optimizer_contains_head_parameters",
                "backbone_parameter_changed_after_step",
                "checkpoint_disk_space_check",
            }
        ]
        result["status"] = "pass" if all(value in {"pass", "none"} for value in required) else "fail"
        return result
    except Exception as exc:
        result["status"] = "fail"
        result["failure_reason"] = str(exc)
        result["elapsed_sec"] = time.time() - started
        return result
    finally:
        try:
            if temp_ckpt.exists():
                temp_ckpt.unlink()
            result["temporary_checkpoint_deleted"] = "pass" if not temp_ckpt.exists() else "fail"
        except Exception as exc:
            result["temporary_checkpoint_deleted"] = "fail"
            result["temporary_checkpoint_delete_error"] = str(exc)
        result["ended_at_utc"] = utc_now()
        write_json(output_path, result)
        if "task_model" in locals():
            try:
                task_model.close()
            except Exception:
                pass
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fresh_lora", "full_ft"], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--benchmark-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--model-dir", default="evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1049)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=2)
    parser.add_argument("--max-val-rows", type=int, default=2)
    args = parser.parse_args()
    payload = run_smoke(args)
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
