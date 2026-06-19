"""
Primary HVUE benchmark evaluator using supervised LoRA finetuning.

Expected CSV columns:
  benchmark,task,split,sequence,label
Optional columns:
  family,group,id

This entrypoint intentionally replaces the previous frozen-feature linear probe
protocol for primary benchmark results. The probe implementation is preserved in
phase2/eval_benchmarks_probe_legacy.py for reference.
"""
import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint
from phase2.lora_utils import (
    PooledEvoClassifier,
    classification_metrics,
    count_total,
    count_trainable,
    encode_labels,
    inject_lora_all_blocks,
    regression_metrics,
    remove_lora_adapters,
)
from phase2.notify import notify
from phase2.utils import tokenize_batch

csv.field_size_limit(sys.maxsize)


DEFAULT_HVUE_FORGET_TASKS = {
    "hvue_human_host_tropism",
    "hvue_human_virus_pathogenicity_cini",
    "hvue_human_virus_pathogenicity_bvbrc_cov",
    "hvue_human_virus_pathogenicity_bvbrc_calici",
    "hvue_human_transmissibility_coronaviridae",
    "hvue_human_transmissibility_orthomyxoviridae",
    "hvue_human_transmissibility_caliciviridae",
}

RESULT_FIELDNAMES = [
    "benchmark",
    "task",
    "group",
    "model_name",
    "checkpoint",
    "seed",
    "problem_type",
    "n_train",
    "n_val",
    "n_test",
    "train_loss",
    "val_loss",
    "validation_metric",
    "metric_for_best",
    "best_step",
    "best_checkpoint",
    "accuracy",
    "f1",
    "auroc",
    "auprc",
    "mse",
    "rmse",
    "r2",
    "pearson",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_modules",
    "trainable_params",
    "total_params",
]


@dataclass
class BenchmarkRecord:
    benchmark: str
    task: str
    split: str
    sequence: str
    label: str
    family: str = ""
    group: str = ""
    record_id: str = ""


def normalize_label(label: str) -> Optional[str]:
    value = str(label).strip()
    if not value or value.lower() in {"nan", "na", "none", "null"}:
        return None
    try:
        numeric = float(value)
    except ValueError:
        return value
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g")


def infer_group(task: str, benchmark: str) -> str:
    if task.lower() in DEFAULT_HVUE_FORGET_TASKS or benchmark.lower() == "hvue":
        return "hvue_forget"
    return "unspecified"


def parse_task_filter(spec: str) -> Optional[set[str]]:
    if not spec:
        return None
    return {part.strip() for part in spec.split(",") if part.strip()}


def read_benchmark_manifest(
    path: str,
    benchmark_scope: str = "hvue",
    task_filter: Optional[set[str]] = None,
    default_benchmark: str = "host_tropism_hiyata",
    default_task: str = "host_tropism_hiyata",
    default_group: str = "host_tropism_adaptation",
) -> List[BenchmarkRecord]:
    records: List[BenchmarkRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"split", "sequence", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Benchmark manifest missing required columns: {sorted(missing)}")
        for row in reader:
            label = normalize_label(row["label"])
            if label is None:
                continue
            benchmark = row.get("benchmark") or default_benchmark
            task = row.get("task") or default_task
            group = row.get("group", "") or infer_group(task, benchmark)
            if group == "unspecified" and "benchmark" not in fields:
                group = default_group
            if task_filter is not None and task not in task_filter:
                continue
            if benchmark_scope == "hvue" and benchmark.lower() != "hvue" and group != "hvue_forget":
                continue
            if benchmark_scope == "task" and task_filter is None:
                raise ValueError("--benchmark-scope task requires --task-filter")
            if benchmark_scope == "task" and task_filter is not None and task not in task_filter:
                continue
            records.append(
                BenchmarkRecord(
                    benchmark=benchmark,
                    task=task,
                    split=row["split"].lower(),
                    sequence=row["sequence"],
                    label=label,
                    family=row.get("family", ""),
                    group=group,
                    record_id=row.get("id") or row.get("record_id", ""),
                )
            )
    return records


def apply_checkpoint(model, ckpt_path: str) -> None:
    delta = load_file(ckpt_path)
    sd = model.state_dict()
    missing = []
    for key, val in delta.items():
        if key not in sd:
            missing.append(key)
            continue
        sd[key].copy_(val.to(sd[key].dtype).to(sd[key].device))
    if missing:
        print(f"[bench-lora] skipped {len(missing)} checkpoint tensors not present in model")
    print(f"[bench-lora] applied {len(delta) - len(missing)} checkpoint tensors from {ckpt_path}")


def tune_runtime(device: str, cpu_threads: int) -> None:
    cpu_threads = max(1, cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(min(4, cpu_threads))
    except RuntimeError:
        pass
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[bench-lora] runtime config: device={device} cpu_threads={cpu_threads}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_rows(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key in RESULT_FIELDNAMES})


def load_existing_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def format_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def summarize(rows: List[dict]) -> Dict[str, dict]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    by_group: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        metric = first_float(row, ["auroc", "f1", "accuracy", "pearson", "r2"])
        if metric is None:
            continue
        by_task[row["task"]].append(metric)
        by_group[row["group"]].append(metric)
    return {
        "groups": {
            group: {"mean_score": float(np.mean(values)), "n_tasks": len(values)}
            for group, values in sorted(by_group.items())
            if values
        },
        "tasks": {
            task: {"mean_score": float(np.mean(values)), "n_rows": len(values)}
            for task, values in sorted(by_task.items())
            if values
        },
    }


def write_summary(path: str, rows: List[dict]) -> None:
    write_json(path, summarize(rows))


def first_float(row: dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            return value_f
    return None


def split_records(records: List[BenchmarkRecord]) -> Dict[str, List[BenchmarkRecord]]:
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        split = "val" if record.split in {"dev", "valid", "validation"} else record.split
        if split in splits:
            splits[split].append(record)
    return splits


def infer_problem_type(labels: List[str], requested: str) -> str:
    if requested != "auto":
        return requested
    return "classification"


def batch_indices(size: int, batch_size: int, shuffle: bool, rng: random.Random) -> Iterable[List[int]]:
    indices = list(range(size))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def trainable_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def load_trainable_state_dict(module: torch.nn.Module, state: Dict[str, torch.Tensor], device: str) -> None:
    named_params = dict(module.named_parameters())
    for name, value in state.items():
        if name not in named_params:
            raise KeyError(f"Checkpoint tensor {name} not found in model")
        param = named_params[name]
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))


def save_best_checkpoint(path: str, module: torch.nn.Module, meta: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": trainable_state_dict(module), "meta": meta}, path)


def select_metric_name(metric_for_best: str, metrics: Dict[str, Optional[float]], problem_type: str) -> str:
    if metric_for_best != "auto":
        return metric_for_best
    if problem_type == "classification":
        for name in ("auroc", "f1", "accuracy"):
            if metrics.get(name) is not None:
                return name
        return "accuracy"
    for name in ("pearson", "r2"):
        if metrics.get(name) is not None:
            return name
    return "mse"


def metric_value_for_selection(
    metric_name: str,
    metrics: Dict[str, Optional[float]],
    val_loss: float,
) -> Optional[float]:
    if metric_name in {"loss", "val_loss"}:
        return -val_loss
    value = metrics.get(metric_name)
    if value is None:
        return None
    if metric_name in {"mse", "rmse"}:
        return -float(value)
    return float(value)


def compute_loss_and_outputs(
    model: PooledEvoClassifier,
    ids: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    problem_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(ids, mask)
    if problem_type == "classification":
        loss = F.cross_entropy(outputs.float(), targets.long())
    else:
        loss = F.mse_loss(outputs.squeeze(-1).float(), targets.float())
    return loss, outputs


def evaluate_model(
    model: PooledEvoClassifier,
    records: List[BenchmarkRecord],
    labels: np.ndarray,
    tokenizer: CharLevelTokenizer,
    batch_size: int,
    max_length: int,
    device: str,
    problem_type: str,
    num_classes: int,
) -> Tuple[float, Dict[str, Optional[float]]]:
    model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    targets_out: List[np.ndarray] = []
    with torch.no_grad():
        for indices in batch_indices(len(records), batch_size, shuffle=False, rng=random.Random(0)):
            batch = [records[idx] for idx in indices]
            ids, mask = tokenize_batch([record.sequence for record in batch], tokenizer, max_length, device)
            target_np = labels[indices]
            if problem_type == "classification":
                targets = torch.tensor(target_np, dtype=torch.long, device=device)
            else:
                targets = torch.tensor(target_np, dtype=torch.float32, device=device)
            loss, outputs = compute_loss_and_outputs(model, ids, mask, targets, problem_type)
            losses.append(float(loss.item()) * len(indices))
            targets_out.append(target_np)
            if problem_type == "classification":
                prob = torch.softmax(outputs.float(), dim=-1).detach().cpu().numpy()
                scores.append(prob)
                preds.append(prob.argmax(axis=1))
            else:
                pred = outputs.squeeze(-1).float().detach().cpu().numpy()
                preds.append(pred)
            del ids, mask, targets, outputs, loss

    denom = max(1, len(records))
    mean_loss = float(sum(losses) / denom)
    y_true = np.concatenate(targets_out, axis=0) if targets_out else np.array([])
    y_pred = np.concatenate(preds, axis=0) if preds else np.array([])
    if problem_type == "classification":
        y_score = np.concatenate(scores, axis=0) if scores else None
        return mean_loss, classification_metrics(y_true, y_pred, y_score, num_classes)
    return mean_loss, regression_metrics(y_true.astype(float), y_pred.astype(float))


def labels_for_split(
    records: List[BenchmarkRecord],
    all_label_to_id: Optional[Dict[str, int]],
    problem_type: str,
) -> np.ndarray:
    if problem_type == "classification":
        assert all_label_to_id is not None
        return np.array([all_label_to_id[record.label] for record in records], dtype=np.int64)
    return np.array([float(record.label) for record in records], dtype=np.float32)


def train_task(
    model,
    task: str,
    task_records: List[BenchmarkRecord],
    tokenizer: CharLevelTokenizer,
    args,
    out_dir: str,
    checkpoint_label: str,
) -> Optional[dict]:
    splits = split_records(task_records)
    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    if n_train == 0 or n_val == 0 or n_test == 0:
        print(f"[bench-lora] skip task={task}: missing train/val/test split")
        return None

    problem_type = infer_problem_type([record.label for record in task_records], args.problem_type)
    label_encoding = None
    if problem_type == "classification":
        _, label_encoding = encode_labels(record.label for record in task_records)
        if label_encoding.num_classes < 2:
            print(f"[bench-lora] skip task={task}: classification needs at least two labels")
            return None
        train_labels = labels_for_split(splits["train"], label_encoding.label_to_id, problem_type)
        val_labels = labels_for_split(splits["val"], label_encoding.label_to_id, problem_type)
        test_labels = labels_for_split(splits["test"], label_encoding.label_to_id, problem_type)
        output_dim = label_encoding.num_classes
    else:
        train_labels = labels_for_split(splits["train"], None, problem_type)
        val_labels = labels_for_split(splits["val"], None, problem_type)
        test_labels = labels_for_split(splits["test"], None, problem_type)
        output_dim = 1

    task_model = None
    optimizer = None
    adapter_params, lora_modules = inject_lora_all_blocks(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    hidden_dim = int(model.blocks[0].pre_norm.scale.shape[0])
    task_model = PooledEvoClassifier(model, hidden_dim, output_dim, problem_type).to(args.device)
    for param in task_model.head.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [param for param in task_model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    task_dir = os.path.join(out_dir, "checkpoints", task)
    best_path = os.path.join(task_dir, "best.pt")
    log_path = os.path.join(out_dir, "logs", f"{task}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    rng = random.Random(args.seed)
    best_score = -float("inf")
    best_payload: Optional[dict] = None
    bad_evals = 0
    global_step = 0
    last_train_loss = float("nan")
    selected_metric_name = args.metric_for_best
    started = time.time()

    try:
        task_model.train()
        with open(log_path, "a") as log_f:
            for epoch in range(1, args.epochs + 1):
                if args.max_steps and global_step >= args.max_steps:
                    break
                for indices in batch_indices(n_train, args.batch_size, shuffle=True, rng=rng):
                    if args.max_steps and global_step >= args.max_steps:
                        break
                    batch = [splits["train"][idx] for idx in indices]
                    target_np = train_labels[indices]
                    ids, mask = tokenize_batch([record.sequence for record in batch], tokenizer, args.max_length, args.device)
                    if problem_type == "classification":
                        targets = torch.tensor(target_np, dtype=torch.long, device=args.device)
                    else:
                        targets = torch.tensor(target_np, dtype=torch.float32, device=args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _outputs = compute_loss_and_outputs(task_model, ids, mask, targets, problem_type)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in task_model.parameters() if param.requires_grad],
                        args.grad_clip,
                    )
                    optimizer.step()
                    global_step += 1
                    last_train_loss = float(loss.item())
                    del ids, mask, targets, loss, _outputs

                    should_eval = args.eval_every > 0 and global_step % args.eval_every == 0
                    if not should_eval:
                        continue
                    val_loss, val_metrics = evaluate_model(
                        task_model,
                        splits["val"],
                        val_labels,
                        tokenizer,
                        args.batch_size,
                        args.max_length,
                        args.device,
                        problem_type,
                        output_dim,
                    )
                    selected_metric_name = select_metric_name(
                        args.metric_for_best,
                        val_metrics,
                        problem_type,
                    )
                    selection_value = metric_value_for_selection(
                        selected_metric_name,
                        val_metrics,
                        val_loss,
                    )
                    improved = selection_value is not None and selection_value > best_score + args.min_delta
                    if improved:
                        best_score = float(selection_value)
                        bad_evals = 0
                        best_payload = {
                            "epoch": epoch,
                            "step": global_step,
                            "train_loss": last_train_loss,
                            "val_loss": val_loss,
                            "val_metrics": val_metrics,
                            "metric_for_best": selected_metric_name,
                            "selection_value": selection_value,
                            "task": task,
                        }
                        save_best_checkpoint(best_path, task_model, best_payload)
                    else:
                        bad_evals += 1

                    log_row = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": last_train_loss,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "metric_for_best": selected_metric_name,
                        "selection_value": selection_value,
                        "best_step": best_payload["step"] if best_payload else "",
                        "bad_evals": bad_evals,
                        "elapsed_sec": time.time() - started,
                    }
                    log_f.write(json.dumps(log_row) + "\n")
                    log_f.flush()
                    print(
                        f"[bench-lora] task={task} step={global_step} "
                        f"train_loss={last_train_loss:.4f} val_loss={val_loss:.4f} "
                        f"{selected_metric_name}={selection_value if selection_value is not None else 'nan'}"
                    )
                    task_model.train()
                    if bad_evals >= args.patience:
                        print(f"[bench-lora] early stop task={task} step={global_step}")
                        break
                if bad_evals >= args.patience:
                    break

            if best_payload is None:
                val_loss, val_metrics = evaluate_model(
                    task_model,
                    splits["val"],
                    val_labels,
                    tokenizer,
                    args.batch_size,
                    args.max_length,
                    args.device,
                    problem_type,
                    output_dim,
                )
                selected_metric_name = select_metric_name(args.metric_for_best, val_metrics, problem_type)
                selection_value = metric_value_for_selection(selected_metric_name, val_metrics, val_loss)
                best_payload = {
                    "epoch": epoch if "epoch" in locals() else 0,
                    "step": global_step,
                    "train_loss": last_train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "metric_for_best": selected_metric_name,
                    "selection_value": selection_value,
                    "task": task,
                }
                save_best_checkpoint(best_path, task_model, best_payload)
                log_row = {
                    "epoch": best_payload["epoch"],
                    "step": global_step,
                    "train_loss": last_train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "metric_for_best": selected_metric_name,
                    "selection_value": selection_value,
                    "best_step": global_step,
                    "bad_evals": bad_evals,
                    "elapsed_sec": time.time() - started,
                }
                log_f.write(json.dumps(log_row) + "\n")
                log_f.flush()

        best_ckpt = torch.load(best_path, map_location="cpu")
        load_trainable_state_dict(task_model, best_ckpt["state_dict"], args.device)
        test_loss, test_metrics = evaluate_model(
            task_model,
            splits["test"],
            test_labels,
            tokenizer,
            args.batch_size,
            args.max_length,
            args.device,
            problem_type,
            output_dim,
        )

        benchmark = task_records[0].benchmark
        explicit_groups = {record.group for record in task_records if record.group}
        group = explicit_groups.pop() if len(explicit_groups) == 1 else infer_group(task, benchmark)
        row = {
            "benchmark": benchmark,
            "task": task,
            "group": group,
            "model_name": args.model_name,
            "checkpoint": checkpoint_label,
            "seed": args.seed,
            "problem_type": problem_type,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "train_loss": best_payload.get("train_loss"),
            "val_loss": best_payload.get("val_loss"),
            "validation_metric": best_payload.get("selection_value"),
            "metric_for_best": best_payload.get("metric_for_best"),
            "best_step": best_payload.get("step"),
            "best_checkpoint": best_path,
            "accuracy": test_metrics.get("accuracy"),
            "f1": test_metrics.get("f1"),
            "auroc": test_metrics.get("auroc"),
            "auprc": test_metrics.get("auprc"),
            "mse": test_metrics.get("mse"),
            "rmse": test_metrics.get("rmse"),
            "r2": test_metrics.get("r2"),
            "pearson": test_metrics.get("pearson"),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_modules": len(lora_modules),
            "trainable_params": count_trainable(task_model),
            "total_params": count_total(task_model),
            "test_loss": test_loss,
        }
        print(
            f"[bench-lora] finished task={task} best_step={row['best_step']} "
            f"test_auroc={row.get('auroc')} test_f1={row.get('f1')} test_acc={row.get('accuracy')}"
        )
        return row
    finally:
        if task_model is not None:
            task_model.close()
        del task_model, optimizer, adapter_params
        remove_lora_adapters(model)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def build_comparison_table(base_dir: str, gd_dir: str, rmu_dir: str, out_csv: str) -> None:
    def read(path: str) -> Dict[Tuple[str, str], float]:
        result = {}
        with open(os.path.join(path, "eval_benchmarks.csv"), newline="") as f:
            for row in csv.DictReader(f):
                for metric in ("accuracy", "f1", "auroc", "auprc", "mse", "rmse", "r2", "pearson"):
                    value = row.get(metric)
                    if value not in (None, ""):
                        result[(row["task"], metric)] = float(value)
        return result

    base = read(base_dir)
    gd = read(gd_dir)
    rmu = read(rmu_dir)
    keys = sorted(set(base) | set(gd) | set(rmu))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "metric", "Original Evo", "GD", "RMU", "delta_GD", "delta_RMU"],
        )
        writer.writeheader()
        for key in keys:
            base_v = base.get(key)
            gd_v = gd.get(key)
            rmu_v = rmu.get(key)
            writer.writerow(
                {
                    "task": key[0],
                    "metric": key[1],
                    "Original Evo": format_cell(base_v),
                    "GD": format_cell(gd_v),
                    "RMU": format_cell(rmu_v),
                    "delta_GD": format_cell(gd_v - base_v if gd_v is not None and base_v is not None else None),
                    "delta_RMU": format_cell(rmu_v - base_v if rmu_v is not None and base_v is not None else None),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Optional unlearned weights.safetensors path")
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument(
        "--benchmark-scope",
        choices=["hvue", "all", "task"],
        default="hvue",
        help="Default hvue preserves primary HVUE filtering; all/task allow local or derived manifests.",
    )
    parser.add_argument(
        "--task-filter",
        default="",
        help="Comma-separated task names to evaluate, e.g. host_tropism_hiyata.",
    )
    parser.add_argument("--default-benchmark", default="host_tropism_hiyata")
    parser.add_argument("--default-task", default="host_tropism_hiyata")
    parser.add_argument("--default-group", default="host_tropism_adaptation")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--model-name", default="Evo-1-8k-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--metric-for-best",
        choices=["auto", "accuracy", "f1", "auroc", "auprc", "mse", "rmse", "r2", "pearson", "loss", "val_loss"],
        default="auto",
    )
    parser.add_argument("--problem-type", choices=["auto", "classification", "regression"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1, help="Task-level progress write interval")
    parser.add_argument("--notify-webhook", default=os.environ.get("FEISHU_WEBHOOK", ""))
    parser.add_argument("--notify-sound", action="store_true")
    parser.add_argument("--notify-on-complete", action="store_true")
    parser.add_argument("--compare-base-dir", default=None)
    parser.add_argument("--compare-gd-dir", default=None)
    parser.add_argument("--compare-rmu-dir", default=None)
    parser.add_argument("--comparison-out", default=None)
    args = parser.parse_args()

    if args.compare_base_dir and args.compare_gd_dir and args.compare_rmu_dir:
        out_csv = args.comparison_out or os.path.join(args.compare_base_dir, "..", "hvue_lora_comparison.csv")
        build_comparison_table(args.compare_base_dir, args.compare_gd_dir, args.compare_rmu_dir, out_csv)
        print(f"[bench-lora] wrote comparison table to {out_csv}")
        return

    set_seed(args.seed)
    tune_runtime(args.device, args.cpu_threads)

    out_dir = args.out_dir or (os.path.dirname(args.ckpt) if args.ckpt else "data/phase2/base_benchmarks")
    results_path = os.path.join(out_dir, "eval_benchmarks.csv")
    summary_path = os.path.join(out_dir, "eval_benchmarks_summary.json")
    progress_path = os.path.join(out_dir, "eval_benchmarks_progress.json")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    if not args.resume and os.path.exists(results_path):
        os.remove(results_path)

    task_filter = parse_task_filter(args.task_filter)
    records = read_benchmark_manifest(
        args.benchmark_manifest,
        benchmark_scope=args.benchmark_scope,
        task_filter=task_filter,
        default_benchmark=args.default_benchmark,
        default_task=args.default_task,
        default_group=args.default_group,
    )
    tasks = defaultdict(list)
    for record in records:
        tasks[record.task].append(record)
    task_items = sorted(tasks.items())
    if not task_items:
        raise RuntimeError(
            f"No rows found in benchmark manifest for scope={args.benchmark_scope} "
            f"task_filter={args.task_filter or '<none>'}"
        )
    print(
        f"[bench-lora] loaded rows={len(records)} tasks={len(task_items)} "
        f"scope={args.benchmark_scope}"
    )

    checkpoint_label = args.ckpt or "base"
    rows: List[dict] = load_existing_rows(results_path) if args.resume else []
    completed = {row["task"] for row in rows if row.get("task")}
    if rows:
        write_summary(summary_path, rows)
        print(f"[bench-lora] resume enabled: loaded {len(rows)} completed task rows")

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if args.ckpt:
        apply_checkpoint(model, args.ckpt)
    else:
        print("[bench-lora] evaluating base model without unlearning checkpoint")
    model.eval()
    tokenizer = CharLevelTokenizer(512)
    started = time.time()
    last_progress: Dict[str, object] = {}

    def report_progress(status: str, **extra: object) -> None:
        nonlocal last_progress
        payload = {
            "status": status,
            "completed_tasks": len(completed),
            "expected_tasks": len(task_items),
            "elapsed_sec": time.time() - started,
        }
        payload.update(extra)
        write_json(progress_path, payload)
        last_progress = payload

    def notify_run(title: str, detail: str, *, force: bool = False) -> None:
        if not force and not args.notify_on_complete:
            return
        notify(
            title=title,
            body=(
                f"checkpoint: {checkpoint_label}\n"
                f"out_dir: {out_dir}\n"
                f"status: {last_progress.get('status', 'unknown')}\n"
                f"task: {last_progress.get('current_task', '')}\n"
                f"completed tasks: {len(completed)}/{len(task_items)}\n"
                f"elapsed min: {(time.time() - started) / 60.0:.1f}\n"
                f"{detail}"
            ),
            webhook_url=args.notify_webhook or None,
            sound=args.notify_sound,
        )

    def handle_signal(signum: int, _frame) -> None:
        signal_name = signal.Signals(signum).name
        report_progress("interrupted", exit_reason=f"received {signal_name}", signal=signal_name)
        notify_run("[bench-lora] interrupted", f"exit_reason: received {signal_name}", force=True)
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled_signal, handle_signal)

    try:
        report_progress("running", phase="initializing")
        for task_idx, (task, task_records) in enumerate(task_items, start=1):
            if task in completed:
                print(f"[bench-lora] skip task {task_idx}/{len(task_items)} task={task}: already complete")
                continue
            report_progress(
                "running",
                phase="train",
                current_task=task,
                task_index=task_idx,
                task_total=len(task_items),
            )
            print(
                f"[bench-lora] task {task_idx}/{len(task_items)} task={task} "
                f"rows={len(task_records)}"
            )
            row = train_task(
                model=model,
                task=task,
                task_records=task_records,
                tokenizer=tokenizer,
                args=args,
                out_dir=out_dir,
                checkpoint_label=checkpoint_label,
            )
            if row is None:
                continue
            rows.append(row)
            append_rows(results_path, [row])
            completed.add(task)
            write_summary(summary_path, rows)
            report_progress(
                "running",
                phase="task_complete",
                current_task=task,
                task_index=task_idx,
                task_total=len(task_items),
            )

        write_summary(summary_path, rows)
        report_progress("complete", phase="complete")
        notify_run("[bench-lora] complete", "exit_reason: complete")
        print(f"[bench-lora] wrote benchmark results to {results_path}")
        print(f"[bench-lora] wrote benchmark summary to {summary_path}")
    except SystemExit:
        raise
    except BaseException as exc:
        report_progress(
            "failed",
            phase=last_progress.get("phase", "exception"),
            current_task=last_progress.get("current_task", ""),
            exit_reason=f"{type(exc).__name__}: {exc}",
        )
        notify_run("[bench-lora] failed", f"exit_reason: {type(exc).__name__}: {exc}", force=True)
        raise


if __name__ == "__main__":
    main()
