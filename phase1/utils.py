import csv
import gzip
import json
import os
import pkgutil
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from stripedhyena.model import StripedHyena
from stripedhyena.utils import dotdict

import evo
from evo.tokenizer import CharLevelTokenizer


def load_local_checkpoint(model_dir: str, config_path: str, device: str | None = None):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    raw_state_dict = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            raw_state_dict.update(load_file(shard_path))
    elif os.path.exists(single_path):
        raw_state_dict = load_file(single_path)
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}. "
            "Expected model.safetensors.index.json or model.safetensors."
        )

    state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("backbone."):
            state_dict[key[len("backbone."):]] = value
        else:
            state_dict[key] = value
    del raw_state_dict

    if "unembed.weight" not in state_dict and "embedding_layer.weight" in state_dict:
        state_dict["unembed.weight"] = state_dict["embedding_layer.weight"]

    config = yaml.safe_load(pkgutil.get_data(evo.__name__, config_path))
    global_config = dotdict(config, Loader=yaml.FullLoader)

    model = StripedHyena(global_config)
    model.load_state_dict(state_dict, strict=True)
    model.to_bfloat16_except_poles_residues()
    if device is not None:
        model = model.to(device)

    return model


@dataclass
class ManifestRecord:
    record_id: str
    label: int
    split: str
    sequence: str
    source: str
    length: int


def read_manifest(path: str) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ManifestRecord(
                    record_id=row["id"],
                    label=int(row["label"]),
                    split=row["split"],
                    sequence=row["sequence"],
                    source=row.get("source", ""),
                    length=int(row.get("length", len(row["sequence"]))),
                )
            )
    return records


def write_manifest(path: str, records: List[ManifestRecord]) -> None:
    fieldnames = ["id", "label", "split", "sequence", "source", "length"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.record_id,
                    "label": record.label,
                    "split": record.split,
                    "sequence": record.sequence,
                    "source": record.source,
                    "length": record.length,
                }
            )


def iter_fasta_gz(path: str) -> Iterable[Tuple[str, str]]:
    header = None
    seq_parts: List[str] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    return re.sub(r"[^ACGTN]", "", seq)


def sample_window(seq: str, max_length: int, rng: random.Random) -> str:
    if len(seq) <= max_length:
        return seq
    start = rng.randint(0, len(seq) - max_length)
    return seq[start : start + max_length]


def pad_batch(token_ids: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(tokens) for tokens in token_ids)
    batch = np.full((len(token_ids), max_len), pad_id, dtype=np.int64)
    mask = np.zeros((len(token_ids), max_len), dtype=np.int64)
    for i, tokens in enumerate(token_ids):
        batch[i, : len(tokens)] = tokens
        mask[i, : len(tokens)] = 1
    return torch.from_numpy(batch), torch.from_numpy(mask)


class FeatureWriter:
    def __init__(self, out_dir: str, num_layers: int, chunk_size: int = 256) -> None:
        self.out_dir = out_dir
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.buffers: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_layers)}
        self.chunk_index: Dict[int, int] = {i: 0 for i in range(num_layers)}
        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_layers):
            os.makedirs(os.path.join(out_dir, f"layer_{i}"), exist_ok=True)

    def add(self, layer_idx: int, features: torch.Tensor) -> None:
        self.buffers[layer_idx].append(features.cpu().numpy())
        if sum(x.shape[0] for x in self.buffers[layer_idx]) >= self.chunk_size:
            self.flush(layer_idx)

    def flush(self, layer_idx: int) -> None:
        if not self.buffers[layer_idx]:
            return
        chunk = np.concatenate(self.buffers[layer_idx], axis=0)
        chunk_path = os.path.join(
            self.out_dir,
            f"layer_{layer_idx}",
            f"chunk_{self.chunk_index[layer_idx]:04d}.npy",
        )
        np.save(chunk_path, chunk)
        self.buffers[layer_idx] = []
        self.chunk_index[layer_idx] += 1

    def flush_all(self) -> None:
        for layer_idx in range(self.num_layers):
            self.flush(layer_idx)
