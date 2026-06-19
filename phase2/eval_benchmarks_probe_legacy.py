"""
Evaluate the base model or an unlearned checkpoint on external HVUE/GUE benchmarks.

Expected CSV columns:
  benchmark,task,split,sequence,label
Optional columns:
  family,group,id

Default benchmark groups:
  - hvue_forget: HVUE human-virus-relevant tasks
  - gue_retain: GUE retain tasks
  - viral_retain: viral retain tasks such as host range / DNA-vs-RNA / HIV type

The evaluator trains a frozen-representation linear probe per task/layer on the
benchmark train split, selects C on validation, and reports test performance.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo.tokenizer import CharLevelTokenizer
from phase2.notify import notify
from phase1.utils import load_local_checkpoint
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

DEFAULT_VIRAL_RETAIN_TASKS = {
    "virus_vs_nonvirus",
    "dna_vs_rna_virus",
    "host_range_prediction",
    "hiv1_vs_hiv2",
    "sars_cov_2_lineage_typing",
    "influenza_subtype_typing",
    "virobench_all_taxon_genus",
    "virobench_all_taxon_times",
    "virobench_dna_taxon_genus",
    "virobench_dna_taxon_times",
    "virobench_rna_taxon_genus",
    "virobench_rna_taxon_times",
}

C_GRID = [0.001, 0.01, 0.1, 1.0]
RESULT_FIELDNAMES = [
    "benchmark",
    "task",
    "group",
    "layer",
    "best_c",
    "selection_score",
    "n_train",
    "n_val",
    "n_test",
    "accuracy",
    "macro_f1",
    "auroc",
    "macro_auroc",
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
        print(f"[bench] skipped {len(missing)} checkpoint tensors not present in model")
    print(f"[bench] applied {len(delta) - len(missing)} checkpoint tensors from {ckpt_path}")


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


def read_benchmark_manifest(path: str) -> List[BenchmarkRecord]:
    records: List[BenchmarkRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"benchmark", "task", "split", "sequence", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Benchmark manifest missing required columns: {sorted(missing)}")
        for row in reader:
            label = normalize_label(row["label"])
            if label is None:
                continue
            records.append(
                BenchmarkRecord(
                    benchmark=row["benchmark"],
                    task=row["task"],
                    split=row["split"].lower(),
                    sequence=row["sequence"],
                    label=label,
                    family=row.get("family", ""),
                    group=row.get("group", ""),
                    record_id=row.get("id", ""),
                )
            )
    return records


def infer_group(task: str, benchmark: str) -> str:
    task_key = task.lower()
    benchmark_key = benchmark.lower()
    if task_key in DEFAULT_HVUE_FORGET_TASKS:
        return "hvue_forget"
    if benchmark_key == "gue" or task_key.startswith("gue_"):
        return "gue_retain"
    if benchmark_key in {"viral_retain", "vgue", "virobench"} or task_key in DEFAULT_VIRAL_RETAIN_TASKS:
        return "viral_retain"
    return "unspecified"


def normalize_label(label: str) -> Optional[str]:
    value = str(label).strip()
    if not value:
        return None
    if value.lower() in {"nan", "na", "none", "null"}:
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


def labels_to_int(labels: Iterable[str]) -> np.ndarray:
    labels = list(labels)
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)


def limit_threads(num_threads: int):
    if threadpool_limits is None:
        class _NullContext:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return _NullContext()
    return threadpool_limits(limits=max(1, num_threads))


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
    print(f"[bench] runtime config: device={device} cpu_threads={cpu_threads}")


def is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def load_existing_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDNAMES})


def write_summary(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summarize(rows), f, indent=2)


def write_progress(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_cache_id(ckpt_path: Optional[str]) -> str:
    if ckpt_path is None:
        return "base"
    return f"{os.path.basename(os.path.dirname(ckpt_path))}_{sha256_file(ckpt_path)[:16]}"


def safe_path_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def task_records_fingerprint(
    task_records: List[BenchmarkRecord],
    layers: List[int],
    max_length: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"eval_benchmarks_features_v1\n")
    digest.update(("layers=" + ",".join(str(layer) for layer in layers) + "\n").encode())
    digest.update(f"max_length={max_length}\n".encode())
    for idx, record in enumerate(task_records):
        digest.update(f"{idx}\t{record.record_id}\t{record.split}\t{record.label}\t".encode())
        digest.update(record.sequence.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:20]


def feature_cache_path(
    cache_dir: Optional[str],
    ckpt_id: str,
    task: str,
    task_fingerprint: str,
) -> Optional[str]:
    if not cache_dir:
        return None
    return os.path.join(
        cache_dir,
        safe_path_component(ckpt_id),
        f"{safe_path_component(task)}_{task_fingerprint}.npz",
    )


def load_feature_cache(path: str, layers: List[int]) -> Optional[Dict[int, np.ndarray]]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            features = {}
            for layer in layers:
                key = f"layer_{layer}"
                if key not in data:
                    return None
                features[layer] = data[key].astype(np.float32, copy=False)
            return features
    except Exception as exc:
        print(f"[bench] ignoring unreadable feature cache {path}: {exc}")
        return None


def save_feature_cache(
    path: str,
    features_by_layer: Dict[int, np.ndarray],
    compression: str = "compressed",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    payload = {f"layer_{layer}": features for layer, features in features_by_layer.items()}
    with open(tmp_path, "wb") as f:
        if compression == "none":
            np.savez(f, **payload)
        else:
            np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)
    print(f"[bench] wrote feature cache {path} compression={compression}")


def get_features_for_task(
    model,
    task_records: List[BenchmarkRecord],
    tokenizer: CharLevelTokenizer,
    layers: List[int],
    batch_size: int,
    auto_batch_size: int,
    max_length: int,
    device: str,
    task_name: str,
    progress_every: int,
    ckpt_id: str,
    feature_cache_dir: Optional[str],
    feature_cache_compression: str,
    feature_cache_write: bool,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[int, np.ndarray]:
    cache_path = feature_cache_path(
        feature_cache_dir,
        ckpt_id,
        task_name,
        task_records_fingerprint(task_records, layers, max_length),
    )
    if cache_path:
        cached = load_feature_cache(cache_path, layers)
        if cached is not None:
            print(f"[bench] feature cache hit task={task_name} path={cache_path}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "cache_hit",
                        "current_task": task_name,
                        "task_sequences_done": len(task_records),
                        "task_sequences_total": len(task_records),
                    }
                )
            return cached

    features = extract_features_for_layers(
        model=model,
        sequences=[record.sequence for record in task_records],
        tokenizer=tokenizer,
        layers=layers,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        max_length=max_length,
        device=device,
        task_name=task_name,
        progress_every=progress_every,
        progress_callback=progress_callback,
    )
    if cache_path and feature_cache_write:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "cache_write",
                    "current_task": task_name,
                    "task_sequences_done": len(task_records),
                    "task_sequences_total": len(task_records),
                    "feature_cache_path": cache_path,
                }
            )
        save_feature_cache(cache_path, features, compression=feature_cache_compression)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "cache_written",
                    "current_task": task_name,
                    "task_sequences_done": len(task_records),
                    "task_sequences_total": len(task_records),
                    "feature_cache_path": cache_path,
                }
            )
    elif cache_path:
        print(f"[bench] feature cache write disabled task={task_name} path={cache_path}")
    return features


def extract_features_for_layers(
    model,
    sequences: List[str],
    tokenizer: CharLevelTokenizer,
    layers: List[int],
    batch_size: int,
    auto_batch_size: int,
    max_length: int,
    device: str,
    task_name: str,
    progress_every: int,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[int, np.ndarray]:
    """Extract mean-pooled activations with length bucketing and batched host copies."""
    num_layers = len(model.blocks)
    layers_set = set(layers)
    feature_buffers: Dict[int, np.ndarray | None] = {layer: None for layer in layers}
    state = {"mask": None, "captured": {}}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            if layer_idx not in layers_set:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx + 1 < num_layers:
                hidden = model.blocks[layer_idx + 1].pre_norm(hidden)
            else:
                hidden = model.norm(hidden)
            mask = state["mask"]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
            state["captured"][layer_idx] = pooled.detach().float()

        return hook

    for layer_idx in layers:
        handles.append(model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

    lengths = np.array([min(len(seq), max_length) for seq in sequences], dtype=np.int32)
    order = np.argsort(lengths, kind="stable")
    current_batch_size = batch_size if batch_size > 0 else max(1, auto_batch_size)
    processed = 0
    batch_count = 0
    next_report = progress_every if progress_every > 0 else len(sequences) + 1
    started = time.time()
    if progress_callback is not None:
        progress_callback({
            "phase": "extract",
            "current_task": task_name,
            "task_sequences_done": 0,
            "task_sequences_total": len(sequences),
            "task_batches": 0,
            "current_batch_size": current_batch_size,
            "task_elapsed_sec": 0.0,
            "task_seq_per_sec": 0.0,
            "task_eta_sec": None,
        })

    try:
        with torch.inference_mode():
            cursor = 0
            while cursor < len(order):
                batch_indices = order[cursor : cursor + current_batch_size]
                batch = [sequences[idx] for idx in batch_indices]
                try:
                    ids, mask = tokenize_batch(batch, tokenizer, max_length, device)
                    state["mask"] = mask
                    state["captured"] = {}
                    _ = model(ids, padding_mask=mask)

                    pooled_stack = torch.stack([state["captured"][layer] for layer in layers], dim=0)
                    pooled_np = pooled_stack.cpu().numpy()
                    for layer_offset, layer_idx in enumerate(layers):
                        layer_feats = pooled_np[layer_offset]
                        if feature_buffers[layer_idx] is None:
                            feature_buffers[layer_idx] = np.empty(
                                (len(sequences), layer_feats.shape[1]),
                                dtype=np.float32,
                            )
                        feature_buffers[layer_idx][batch_indices] = layer_feats

                    processed += len(batch_indices)
                    batch_count += 1
                    cursor += len(batch_indices)
                    if processed >= next_report or cursor == len(order):
                        elapsed = max(time.time() - started, 1e-6)
                        seq_per_sec = processed / elapsed
                        remaining = max(len(sequences) - processed, 0)
                        eta_sec = remaining / max(seq_per_sec, 1e-6)
                        payload = {
                            "phase": "extract",
                            "current_task": task_name,
                            "task_sequences_done": processed,
                            "task_sequences_total": len(sequences),
                            "task_batches": batch_count,
                            "current_batch_size": current_batch_size,
                            "task_elapsed_sec": elapsed,
                            "task_seq_per_sec": seq_per_sec,
                            "task_eta_sec": eta_sec,
                        }
                        if progress_callback is not None:
                            progress_callback(payload)
                        print(
                            f"[bench] extract task={task_name} "
                            f"done={processed}/{len(sequences)} "
                            f"batches={batch_count} "
                            f"batch_size={current_batch_size} "
                            f"seq_per_sec={seq_per_sec:.1f} "
                            f"eta_min={eta_sec / 60.0:.1f}"
                        )
                        next_report += progress_every
                    del ids, mask, pooled_stack, pooled_np
                except RuntimeError as exc:
                    if device.startswith("cuda") and is_oom_error(exc) and current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        torch.cuda.empty_cache()
                        print(
                            f"[bench] OOM during extraction for task={task_name}; "
                            f"batch_size {current_batch_size} -> {new_batch_size}"
                        )
                        current_batch_size = new_batch_size
                        continue
                    raise
    finally:
        for handle in handles:
            handle.remove()

    missing_layers = [layer for layer, feats in feature_buffers.items() if feats is None]
    if missing_layers:
        raise RuntimeError(f"No features captured for layers: {missing_layers}")
    return {layer: feature_buffers[layer] for layer in layers}


def score_classifier(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = model.predict(x)
    scores = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    classes = np.asarray(model.classes_)
    observed = np.unique(y)
    if len(classes) == 2 and len(observed) == 2:
        prob = model.predict_proba(x)[:, 1]
        scores["auroc"] = float(roc_auc_score(y, prob))
    elif len(classes) > 2 and set(observed.tolist()) == set(classes.tolist()):
        prob = model.predict_proba(x)
        scores["macro_auroc"] = float(
            roc_auc_score(y, prob, multi_class="ovr", average="macro")
        )
    return scores


def select_metric(scores: Dict[str, float]) -> float:
    if "auroc" in scores:
        return scores["auroc"]
    if "macro_auroc" in scores:
        return scores["macro_auroc"]
    return scores["accuracy"]


def resolve_solver(solver: str, n_train: int) -> Tuple[str, int, float]:
    if solver == "auto":
        if n_train >= 50000:
            return "saga", 200, 1e-3
        return "lbfgs", 1000, 1e-4
    if solver == "saga":
        return "saga", 200, 1e-3
    return "lbfgs", 1000, 1e-4


def train_probe_for_task(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    solver: str,
) -> Dict[str, object] | None:
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None
    if len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[test_mask])) < 2:
        return None

    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_mask])
    x_test = scaler.transform(features[test_mask])
    x_val = scaler.transform(features[val_mask]) if val_mask.sum() else None

    resolved_solver, max_iter, tol = resolve_solver(solver, int(train_mask.sum()))
    best_model = None
    best_c = None
    best_score = -np.inf
    for c in C_GRID:
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=max_iter,
            solver=resolved_solver,
            tol=tol,
        )
        clf.fit(x_train, labels[train_mask])
        if x_val is not None and len(np.unique(labels[val_mask])) >= 2:
            val_scores = score_classifier(clf, x_val, labels[val_mask])
            score = select_metric(val_scores)
        else:
            score = score_classifier(clf, x_train, labels[train_mask])["accuracy"]
        if score > best_score:
            best_model = clf
            best_c = c
            best_score = score

    assert best_model is not None
    test_scores = score_classifier(best_model, x_test, labels[test_mask])
    return {
        "best_c": best_c,
        "selection_score": best_score,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        **test_scores,
    }


def fit_task_layers(
    pending_layers: List[int],
    features_by_layer: Dict[int, np.ndarray],
    labels: np.ndarray,
    splits: np.ndarray,
    probe_jobs: int,
    cpu_threads: int,
    solver: str,
) -> List[Tuple[int, Dict[str, object] | None]]:
    def run_one(layer: int) -> Tuple[int, Dict[str, object] | None]:
        return layer, train_probe_for_task(features_by_layer[layer], labels, splits, solver)

    n_jobs = max(1, min(probe_jobs, len(pending_layers)))
    if Parallel is None or delayed is None or n_jobs == 1:
        results = []
        with limit_threads(cpu_threads):
            for layer in pending_layers:
                results.append(run_one(layer))
        return results

    with limit_threads(1):
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_one)(layer) for layer in pending_layers
        )


def summarize(rows: List[dict]) -> Dict[str, dict]:
    by_group = defaultdict(list)
    by_benchmark = defaultdict(list)
    by_task = defaultdict(list)
    for row in rows:
        metric = None
        for metric_name in ("auroc", "macro_auroc", "accuracy"):
            raw_metric = row.get(metric_name)
            if raw_metric in (None, ""):
                continue
            value = float(raw_metric)
            if not np.isnan(value):
                metric = value
                break
        if metric is not None:
            by_group[row["group"]].append(metric)
            by_benchmark[row["benchmark"]].append(metric)
            by_task[row["task"]].append(metric)
    return {
        "groups": {
            group: {"mean_score": float(np.mean(values)), "n_task_layers": len(values)}
            for group, values in sorted(by_group.items())
            if values
        },
        "benchmarks": {
            benchmark: {"mean_score": float(np.mean(values)), "n_task_layers": len(values)}
            for benchmark, values in sorted(by_benchmark.items())
            if values
        },
        "tasks": {
            task: {"mean_score": float(np.mean(values)), "n_layers": len(values)}
            for task, values in sorted(by_task.items())
            if values
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to unlearned weights.safetensors. Omit for base-model evaluation.",
    )
    parser.add_argument(
        "--benchmark-manifest",
        required=True,
        help="CSV with benchmark,task,split,sequence,label columns",
    )
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Feature-extraction batch size. Use 0 to start high and back off on OOM.",
    )
    parser.add_argument(
        "--auto-batch-size",
        type=int,
        default=64,
        help="Starting batch size when --batch-size=0.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(os.cpu_count() or 1, 16),
        help="CPU threads budget for probe fitting.",
    )
    parser.add_argument(
        "--probe-jobs",
        type=int,
        default=7,
        help="Parallel linear-probe fits across layers.",
    )
    parser.add_argument(
        "--probe-solver",
        choices=["auto", "lbfgs", "saga"],
        default="auto",
        help="Linear probe solver. auto uses saga for very large train splits.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1024,
        help="Extraction progress report interval in sequences.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing eval_benchmarks.csv in --out-dir.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--layers",
        default="3-9",
        help="Comma list/ranges used for frozen representation probes",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to the checkpoint directory",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Optional directory for checkpoint/task/layer activation caches.",
    )
    parser.add_argument(
        "--feature-cache-compression",
        choices=["compressed", "none"],
        default="compressed",
        help="Feature cache write format. Existing caches are readable either way.",
    )
    parser.add_argument(
        "--no-feature-cache-write",
        action="store_true",
        help="Read existing feature caches but do not write new ones.",
    )
    parser.add_argument(
        "--notify-webhook",
        default=os.environ.get("FEISHU_WEBHOOK", ""),
        help="Optional Feishu incoming webhook URL. Defaults to FEISHU_WEBHOOK.",
    )
    parser.add_argument(
        "--notify-sound",
        action="store_true",
        help="Emit a terminal bell when the run completes or exits early.",
    )
    parser.add_argument(
        "--notify-on-complete",
        action="store_true",
        help="Also send notifications for successful completion.",
    )
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    out_dir = args.out_dir or (os.path.dirname(args.ckpt) if args.ckpt else "data/phase2/base_benchmarks")
    results_path = os.path.join(out_dir, "eval_benchmarks.csv")
    summary_path = os.path.join(out_dir, "eval_benchmarks_summary.json")
    progress_path = os.path.join(out_dir, "eval_benchmarks_progress.json")
    os.makedirs(out_dir, exist_ok=True)

    if not args.resume and os.path.exists(results_path):
        os.remove(results_path)

    tune_runtime(args.device, args.cpu_threads)
    records = read_benchmark_manifest(args.benchmark_manifest)
    tasks = defaultdict(list)
    for record in records:
        tasks[record.task].append(record)
    task_items = sorted(tasks.items())
    expected_task_layers = len(task_items) * len(layers)
    print(
        f"[bench] loaded manifest rows={len(records)} tasks={len(task_items)} "
        f"layers={layers} expected_task_layers={expected_task_layers}"
    )

    ckpt_id = checkpoint_cache_id(args.ckpt) if args.feature_cache_dir else ""
    if args.feature_cache_dir:
        print(f"[bench] feature cache enabled dir={args.feature_cache_dir} ckpt_id={ckpt_id}")

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if args.ckpt:
        apply_checkpoint(model, args.ckpt)
    else:
        print("[bench] evaluating base model without unlearning checkpoint")
    model.eval()
    tokenizer = CharLevelTokenizer(512)

    rows: List[dict] = load_existing_rows(results_path) if args.resume else []
    completed = {
        (row["task"], int(row["layer"]))
        for row in rows
        if row.get("task") and row.get("layer") not in (None, "")
    }
    if rows:
        print(f"[bench] resume enabled: loaded {len(rows)} existing task-layer rows")
        write_summary(summary_path, rows)

    overall_started = time.time()
    last_progress: Dict[str, object] = {}

    def report_progress(status: str, **extra: object) -> None:
        nonlocal last_progress
        payload = {
            "status": status,
            "task_index": int(extra.pop("task_index", 0)),
            "task_total": len(task_items),
            "completed_task_layers": len(completed),
            "expected_task_layers": expected_task_layers,
            "elapsed_sec": time.time() - overall_started,
        }
        payload.update(extra)
        write_progress(progress_path, payload)
        last_progress = payload

    def notify_run(title: str, detail: str, *, force: bool = False) -> None:
        if not force and not args.notify_on_complete:
            return
        ckpt_label = args.ckpt or "base"
        current_task = last_progress.get("current_task", "")
        phase = last_progress.get("phase", "")
        body = (
            f"checkpoint: {ckpt_label}\n"
            f"out_dir: {out_dir}\n"
            f"status: {last_progress.get('status', 'unknown')}\n"
            f"task: {last_progress.get('task_index', 0)}/{len(task_items)} {current_task}\n"
            f"phase: {phase}\n"
            f"completed layers: {len(completed)}/{expected_task_layers}\n"
            f"elapsed min: {(time.time() - overall_started) / 60.0:.1f}\n"
            f"{detail}"
        )
        notify(
            title=title,
            body=body,
            webhook_url=args.notify_webhook or None,
            sound=args.notify_sound,
        )

    def handle_signal(signum: int, _frame) -> None:
        signal_name = signal.Signals(signum).name
        report_progress(
            "interrupted",
            task_index=int(last_progress.get("task_index", 0)),
            phase=last_progress.get("phase", "signal"),
            current_task=last_progress.get("current_task", ""),
            exit_reason=f"received {signal_name}",
            signal=signal_name,
        )
        notify_run(
            "[bench] interrupted",
            f"exit_reason: received {signal_name}",
            force=True,
        )
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled_signal, handle_signal)

    try:
        report_progress("running", phase="initializing", task_index=0)

        for task_idx, (task, task_records) in enumerate(task_items, start=1):
            labels = labels_to_int(record.label for record in task_records)
            splits = np.array([record.split for record in task_records])
            benchmark = task_records[0].benchmark
            explicit_groups = {record.group for record in task_records if record.group}
            group = explicit_groups.pop() if len(explicit_groups) == 1 else infer_group(task, benchmark)
            pending_layers = [layer for layer in layers if (task, layer) not in completed]

            if not pending_layers:
                print(f"[bench] skip task {task_idx}/{len(task_items)} task={task}: already complete")
                report_progress("running", current_task=task, task_index=task_idx, phase="skip")
                continue

            print(
                f"[bench] task {task_idx}/{len(task_items)} extracting task={task} "
                f"n={len(task_records)} group={group} layers={pending_layers}"
            )
            task_started = time.time()
            report_progress(
                "running",
                current_task=task,
                task_index=task_idx,
                phase="extract",
                task_sequences_done=0,
                task_sequences_total=len(task_records),
                current_batch_size=args.batch_size if args.batch_size > 0 else args.auto_batch_size,
            )
            features_by_layer = get_features_for_task(
                model=model,
                task_records=task_records,
                tokenizer=tokenizer,
                layers=pending_layers,
                batch_size=args.batch_size,
                auto_batch_size=args.auto_batch_size,
                max_length=args.max_length,
                device=args.device,
                task_name=task,
                progress_every=args.progress_every,
                ckpt_id=ckpt_id,
                feature_cache_dir=args.feature_cache_dir,
                feature_cache_compression=args.feature_cache_compression,
                feature_cache_write=not args.no_feature_cache_write,
                progress_callback=lambda payload: report_progress(
                    "running",
                    task_index=task_idx,
                    **payload,
                ),
            )
            report_progress(
                "running",
                current_task=task,
                task_index=task_idx,
                phase="fit",
                task_sequences_done=len(task_records),
                task_sequences_total=len(task_records),
            )
            layer_results = fit_task_layers(
                pending_layers=pending_layers,
                features_by_layer=features_by_layer,
                labels=labels,
                splits=splits,
                probe_jobs=args.probe_jobs,
                cpu_threads=args.cpu_threads,
                solver=args.probe_solver,
            )

            task_rows: List[dict] = []
            for layer, result in layer_results:
                if result is None:
                    print(f"[bench] skipped task={task} layer={layer}: insufficient splits/classes")
                    continue
                row = {
                    "benchmark": benchmark,
                    "task": task,
                    "group": group,
                    "layer": layer,
                    **result,
                }
                rows.append(row)
                task_rows.append(row)
                completed.add((task, layer))
                metric = next(
                    value
                    for value in (row.get("auroc"), row.get("macro_auroc"), row.get("accuracy"))
                    if value is not None and not np.isnan(float(value))
                )
                print(f"  layer {layer:>2}: score={metric:.4f} acc={row['accuracy']:.4f}")

            append_rows(results_path, task_rows)
            write_summary(summary_path, rows)
            report_progress(
                "running",
                current_task=task,
                task_index=task_idx,
                phase="task_complete",
                task_sequences_done=len(task_records),
                task_sequences_total=len(task_records),
                wrote_rows=len(task_rows),
                task_minutes=(time.time() - task_started) / 60.0,
            )
            print(
                f"[bench] finished task={task} wrote_rows={len(task_rows)} "
                f"task_minutes={(time.time() - task_started) / 60.0:.2f} "
                f"total_rows={len(rows)}"
            )
            del features_by_layer
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        write_summary(summary_path, rows)
        report_progress("complete", phase="complete", task_index=len(task_items))
        notify_run("[bench] complete", "exit_reason: complete")
        print(f"[bench] wrote benchmark results to {results_path}")
        print(f"[bench] wrote benchmark summary to {summary_path}")
    except SystemExit:
        raise
    except BaseException as exc:
        report_progress(
            "failed",
            task_index=int(last_progress.get("task_index", 0)),
            phase=last_progress.get("phase", "exception"),
            current_task=last_progress.get("current_task", ""),
            exit_reason=f"{type(exc).__name__}: {exc}",
        )
        notify_run(
            "[bench] failed",
            f"exit_reason: {type(exc).__name__}: {exc}",
            force=True,
        )
        raise


if __name__ == "__main__":
    main()
