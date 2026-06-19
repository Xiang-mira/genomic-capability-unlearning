"""
Prepare a unified HVUE/GUE/viral-retain benchmark manifest for Phase 2 evaluation.

This script is designed for restartable data preparation after interrupted
downloads. It supports:
  1. Reusing existing raw benchmark directories.
  2. Downloading HVUE task CSVs from the official Hugging Face dataset repo.
  3. Incorporating GUE task directories from an extracted local archive.
  4. Incorporating task-ready viral-retain tables when present locally.

The output manifest contains:
  benchmark,task,split,sequence,label,family,group,id
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

csv.field_size_limit(sys.maxsize)


HVUE_FILES = {
    "Host_Tropism/train.csv": ("hvue_human_host_tropism", "", "hvue_forget"),
    "Host_Tropism/dev.csv": ("hvue_human_host_tropism", "", "hvue_forget"),
    "Host_Tropism/test.csv": ("hvue_human_host_tropism", "", "hvue_forget"),
    "Pathogenecity/CINI/train.csv": ("hvue_human_virus_pathogenicity_cini", "mixed", "hvue_forget"),
    "Pathogenecity/CINI/dev.csv": ("hvue_human_virus_pathogenicity_cini", "mixed", "hvue_forget"),
    "Pathogenecity/CINI/test.csv": ("hvue_human_virus_pathogenicity_cini", "mixed", "hvue_forget"),
    "Pathogenecity/BVBRC_CoV/train.csv": ("hvue_human_virus_pathogenicity_bvbrc_cov", "Coronaviridae", "hvue_forget"),
    "Pathogenecity/BVBRC_CoV/dev.csv": ("hvue_human_virus_pathogenicity_bvbrc_cov", "Coronaviridae", "hvue_forget"),
    "Pathogenecity/BVBRC_CoV/test.csv": ("hvue_human_virus_pathogenicity_bvbrc_cov", "Coronaviridae", "hvue_forget"),
    "Pathogenecity/BVBRC_Calci/train.csv": ("hvue_human_virus_pathogenicity_bvbrc_calici", "Caliciviridae", "hvue_forget"),
    "Pathogenecity/BVBRC_Calci/dev.csv": ("hvue_human_virus_pathogenicity_bvbrc_calici", "Caliciviridae", "hvue_forget"),
    "Pathogenecity/BVBRC_Calci/test.csv": ("hvue_human_virus_pathogenicity_bvbrc_calici", "Caliciviridae", "hvue_forget"),
    "Transmissibility/Coronovirdae/train.csv": ("hvue_human_transmissibility_coronaviridae", "Coronaviridae", "hvue_forget"),
    "Transmissibility/Coronovirdae/dev.csv": ("hvue_human_transmissibility_coronaviridae", "Coronaviridae", "hvue_forget"),
    "Transmissibility/Coronovirdae/test.csv": ("hvue_human_transmissibility_coronaviridae", "Coronaviridae", "hvue_forget"),
    "Transmissibility/Orthomyxovirdae/train.csv": ("hvue_human_transmissibility_orthomyxoviridae", "Orthomyxoviridae", "hvue_forget"),
    "Transmissibility/Orthomyxovirdae/dev.csv": ("hvue_human_transmissibility_orthomyxoviridae", "Orthomyxoviridae", "hvue_forget"),
    "Transmissibility/Orthomyxovirdae/test.csv": ("hvue_human_transmissibility_orthomyxoviridae", "Orthomyxoviridae", "hvue_forget"),
    "Transmissibility/Calcivirdae/train.csv": ("hvue_human_transmissibility_caliciviridae", "Caliciviridae", "hvue_forget"),
    "Transmissibility/Calcivirdae/dev.csv": ("hvue_human_transmissibility_caliciviridae", "Caliciviridae", "hvue_forget"),
    "Transmissibility/Calcivirdae/test.csv": ("hvue_human_transmissibility_caliciviridae", "Caliciviridae", "hvue_forget"),
}

GUE_TASK_DIRS = [
    "GUE/prom_300_all",
    "GUE/prom_300_notata",
    "GUE/prom_300_tata",
    "GUE/prom_core_all",
    "GUE/prom_core_notata",
    "GUE/prom_core_tata",
    "GUE/splice_reconstructed",
    "GUE/human_tf_0",
    "GUE/human_tf_1",
    "GUE/human_tf_2",
    "GUE/human_tf_3",
    "GUE/human_tf_4",
    "GUE/mouse_0",
    "GUE/mouse_1",
    "GUE/mouse_2",
    "GUE/mouse_3",
    "GUE/mouse_4",
    "GUE/EPI_GM12878",
    "GUE/EPI_HUVEC",
    "GUE/EPI_HeLa-S3",
    "GUE/EPI_IMR90",
    "GUE/EPI_K562",
    "GUE/EPI_NHEK",
    "GUE/emp_H3",
    "GUE/emp_H3K14ac",
    "GUE/emp_H3K36me3",
    "GUE/emp_H3K4me1",
    "GUE/emp_H3K4me2",
    "GUE/emp_H3K4me3",
    "GUE/emp_H3K79me3",
    "GUE/emp_H3K9ac",
    "GUE/emp_H4",
    "GUE/emp_H4ac",
]

DEFAULT_VIRAL_RETAIN_TASKS = [
    "virus_vs_nonvirus",
    "dna_vs_rna_virus",
    "host_range_prediction",
    "hiv1_vs_hiv2",
    "sars_cov_2_lineage_typing",
    "influenza_subtype_typing",
]

VIROBENCH_REPO_ID = "YDXX/ViroBench"
VIROBENCH_DATASET_DIR = "ViroBench-CLS-Lite"
VIROBENCH_NA_TYPES = ("ALL", "DNA", "RNA")
VIROBENCH_TASK_KINDS = ("taxon", "host")
VIROBENCH_SPLIT_MODES = ("genus", "times")
DEFAULT_VIROBENCH_RETAIN_TASKS = [
    "virobench_all_taxon_genus",
    "virobench_all_taxon_times",
    "virobench_dna_taxon_genus",
    "virobench_dna_taxon_times",
    "virobench_rna_taxon_genus",
    "virobench_rna_taxon_times",
]
VIROBENCH_TASK_SPECS = {
    f"virobench_{na_type.lower()}_{task_kind}_{split_mode}": {
        "na_type": na_type,
        "task_kind": task_kind,
        "split_mode": split_mode,
        "label_col": "family" if task_kind == "taxon" else "host_label",
    }
    for na_type in VIROBENCH_NA_TYPES
    for task_kind in VIROBENCH_TASK_KINDS
    for split_mode in VIROBENCH_SPLIT_MODES
}

VGUE_TASK_ALIASES = {
    "virus_vs_nonvirus": "virus_vs_nonvirus",
    "virus_vs_non_virus": "virus_vs_nonvirus",
    "virus_vs_non-virus": "virus_vs_nonvirus",
    "viral_vs_nonviral": "virus_vs_nonvirus",
    "viral_vs_non_viral": "virus_vs_nonvirus",
    "dna_vs_rna": "dna_vs_rna_virus",
    "dna_vs_rna_virus": "dna_vs_rna_virus",
    "dna_rna": "dna_vs_rna_virus",
    "host_prediction": "host_range_prediction",
    "host_range": "host_range_prediction",
    "host_range_prediction": "host_range_prediction",
    "host_prediction_range": "host_range_prediction",
    "hiv1_vs_hiv2": "hiv1_vs_hiv2",
    "hiv_1_vs_hiv_2": "hiv1_vs_hiv2",
    "hiv-1_vs_hiv-2": "hiv1_vs_hiv2",
    "sars_cov_2_lineage": "sars_cov_2_lineage_typing",
    "sars_cov_2_lineage_typing": "sars_cov_2_lineage_typing",
    "sars_cov2_lineage": "sars_cov_2_lineage_typing",
    "sars_cov2_lineage_typing": "sars_cov_2_lineage_typing",
    "covid_lineage": "sars_cov_2_lineage_typing",
    "influenza_subtype": "influenza_subtype_typing",
    "influenza_subtype_typing": "influenza_subtype_typing",
    "influenza_a_subtype": "influenza_subtype_typing",
    "hiv1_tropism": "hiv1_tropism",
    "hiv_1_tropism": "hiv1_tropism",
    "hiv_tropism": "hiv1_tropism",
}

TASK_COLUMN_CANDIDATES = ("task", "benchmark_task", "dataset", "dataset_name", "name")
SPLIT_COLUMN_CANDIDATES = ("split", "partition", "subset")

SPLIT_MAP = {"dev": "val", "valid": "val", "validation": "val"}


@dataclass
class ManifestRow:
    benchmark: str
    task: str
    split: str
    sequence: str
    label: str
    family: str
    group: str
    row_id: str


def normalize_split(name: str) -> str:
    key = name.strip().lower()
    return SPLIT_MAP.get(key, key)


def normalize_task_name(name: object) -> str:
    key = str(name).strip().lower()
    key = key.replace("/", "_").replace("\\", "_")
    key = key.replace(" ", "_").replace("-", "_")
    key = "_".join(part for part in key.split("_") if part)
    return VGUE_TASK_ALIASES.get(key, key)


def infer_task_column(df: pd.DataFrame) -> Optional[str]:
    for col in TASK_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def infer_split_column(df: pd.DataFrame) -> Optional[str]:
    for col in SPLIT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return "".join(ch for ch in seq if ch in {"A", "C", "G", "T", "U", "N"})


def normalize_label(raw_label: object) -> Optional[str]:
    if pd.isna(raw_label):
        return None
    label = str(raw_label).strip()
    if not label:
        return None
    if label.lower() in {"nan", "na", "none", "null"}:
        return None
    try:
        value = float(label)
    except ValueError:
        return label
    if not math.isfinite(value):
        return None
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def download_hvue(raw_root: Path) -> List[str]:
    downloaded = []
    total = len(HVUE_FILES)
    for idx, rel_path in enumerate(HVUE_FILES, start=1):
        local_path = raw_root / "hvue" / rel_path
        if local_path.exists():
            print(f"[prepare_benchmarks] HVUE {idx}/{total} already present: {rel_path}")
            continue
        ensure_parent(local_path)
        print(f"[prepare_benchmarks] HVUE {idx}/{total} downloading: {rel_path}")
        tmp = hf_hub_download(
            repo_id="duttaprat/HVUE",
            repo_type="dataset",
            filename=rel_path,
        )
        shutil.copy2(tmp, local_path)
        print(f"[prepare_benchmarks] HVUE {idx}/{total} done: {rel_path}")
        downloaded.append(str(local_path))
    return downloaded


def download_gue(raw_root: Path) -> List[str]:
    downloaded = []
    total = len(GUE_TASK_DIRS) * 3
    step = 0
    for task_dir in GUE_TASK_DIRS:
        for split in ("train", "dev", "test"):
            step += 1
            rel_path = f"{task_dir}/{split}.csv"
            local_path = raw_root / "gue" / rel_path
            if local_path.exists():
                print(f"[prepare_benchmarks] GUE {step}/{total} already present: {rel_path}")
                continue
            ensure_parent(local_path)
            print(f"[prepare_benchmarks] GUE {step}/{total} downloading: {rel_path}")
            tmp = hf_hub_download(
                repo_id="leannmlindsey/GUE",
                repo_type="dataset",
                filename=rel_path,
            )
            shutil.copy2(tmp, local_path)
            print(f"[prepare_benchmarks] GUE {step}/{total} done: {rel_path}")
            downloaded.append(str(local_path))
    return downloaded


def virobench_rel_path(task_name: str, split: str, suffix: str) -> str:
    spec = VIROBENCH_TASK_SPECS[task_name]
    filename = f"{split}_sequences{suffix}" if suffix == ".jsonl" else f"{split}{suffix}"
    return (
        f"{VIROBENCH_DATASET_DIR}/{spec['na_type']}/{spec['task_kind']}/"
        f"{spec['split_mode']}/{filename}"
    )


def download_virobench(raw_root: Path, tasks: Iterable[str]) -> List[str]:
    downloaded = []
    tasks = list(tasks)
    total = len(tasks) * 3 * 2
    step = 0
    for task_name in tasks:
        if task_name not in VIROBENCH_TASK_SPECS:
            raise ValueError(f"Unknown ViroBench task: {task_name}")
        for split in ("train", "val", "test"):
            for suffix in (".csv", ".jsonl"):
                step += 1
                rel_path = virobench_rel_path(task_name, split, suffix)
                local_path = raw_root / "virobench" / rel_path
                if local_path.exists():
                    print(f"[prepare_benchmarks] ViroBench {step}/{total} already present: {rel_path}")
                    continue
                ensure_parent(local_path)
                print(f"[prepare_benchmarks] ViroBench {step}/{total} downloading: {rel_path}")
                tmp = hf_hub_download(
                    repo_id=VIROBENCH_REPO_ID,
                    repo_type="dataset",
                    filename=rel_path,
                )
                shutil.copy2(tmp, local_path)
                print(f"[prepare_benchmarks] ViroBench {step}/{total} done: {rel_path}")
                downloaded.append(str(local_path))
    return downloaded


def collect_hvue_rows(raw_root: Path) -> tuple[List[ManifestRow], List[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    total = len(HVUE_FILES)
    for file_idx, (rel_path, (task, family, group)) in enumerate(sorted(HVUE_FILES.items()), start=1):
        csv_path = raw_root / "hvue" / rel_path
        if not csv_path.exists():
            missing.append(str(csv_path))
            print(f"[prepare_benchmarks] HVUE build {file_idx}/{total} missing: {rel_path}")
            continue
        split = normalize_split(csv_path.stem)
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError(f"HVUE file missing required columns: {csv_path}")
        before = len(rows)
        skipped = 0
        for row_idx, record in df.iterrows():
            seq = clean_sequence(record["sequence"])
            if not seq:
                continue
            label = normalize_label(record["label"])
            if label is None:
                skipped += 1
                continue
            rows.append(
                ManifestRow(
                    benchmark="hvue",
                    task=task,
                    split=split,
                    sequence=seq,
                    label=label,
                    family=family,
                    group=group,
                    row_id=f"hvue|{task}|{split}|{row_idx}",
                )
            )
        print(
            f"[prepare_benchmarks] HVUE build {file_idx}/{total} loaded {len(rows) - before} rows "
            f"from {rel_path} skipped={skipped} (total={len(rows)})"
        )
    return rows, missing


def infer_sequence_column(df: pd.DataFrame) -> str:
    for col in ("sequence", "seq", "text"):
        if col in df.columns:
            return col
    raise ValueError(f"Could not find sequence column in columns={list(df.columns)}")


def build_sequence_series(df: pd.DataFrame) -> pd.Series:
    if "sequence" in df.columns:
        return df["sequence"]
    if "seq" in df.columns:
        return df["seq"]
    if "text" in df.columns:
        return df["text"]
    if "enhancer" in df.columns and "promoter" in df.columns:
        return df["enhancer"].astype(str) + "NNNNNNNNNN" + df["promoter"].astype(str)
    raise ValueError(f"Could not derive sequence input from columns={list(df.columns)}")


def infer_label_column(df: pd.DataFrame) -> str:
    for col in ("label", "labels", "target"):
        if col in df.columns:
            return col
    raise ValueError(f"Could not find label column in columns={list(df.columns)}")


def collect_gue_rows(raw_root: Path) -> tuple[List[ManifestRow], List[str]]:
    gue_root = raw_root / "gue" / "GUE"
    if not gue_root.exists():
        return [], [str(gue_root)]
    rows: List[ManifestRow] = []
    missing: List[str] = []
    total = len(GUE_TASK_DIRS)
    for task_idx, rel_task_dir in enumerate(GUE_TASK_DIRS, start=1):
        task_dir = raw_root / "gue" / rel_task_dir
        task_name = f"gue_{Path(rel_task_dir).name.lower().replace('-', '_')}"
        if not task_dir.exists():
            missing.append(str(task_dir))
            print(f"[prepare_benchmarks] GUE build {task_idx}/{total} missing dir: {rel_task_dir}")
            continue
        seen_split = False
        for csv_path in sorted(task_dir.glob("*.csv")):
            split = normalize_split(csv_path.stem)
            if split not in {"train", "val", "test"}:
                continue
            seen_split = True
            df = pd.read_csv(csv_path)
            label_col = infer_label_column(df)
            seq_series = build_sequence_series(df)
            before = len(rows)
            skipped = 0
            for idx, record in df.iterrows():
                seq = clean_sequence(seq_series.iloc[idx])
                if not seq:
                    continue
                label = normalize_label(record[label_col])
                if label is None:
                    skipped += 1
                    continue
                rows.append(
                    ManifestRow(
                        benchmark="gue",
                        task=task_name,
                        split=split,
                        sequence=seq,
                        label=label,
                        family="",
                        group="gue_retain",
                        row_id=f"gue|{task_name}|{split}|{idx}",
                    )
                )
            print(
                f"[prepare_benchmarks] GUE build {task_idx}/{total} loaded {len(rows) - before} rows "
                f"from {csv_path.relative_to(raw_root)} skipped={skipped} (total={len(rows)})"
            )
        if not seen_split:
            missing.append(str(task_dir))
            print(f"[prepare_benchmarks] GUE build {task_idx}/{total} no usable splits: {rel_task_dir}")
    return rows, missing


def read_virobench_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            taxid = str(payload.get("taxid", "")).strip()
            parts = payload.get("sequences", [])
            if not taxid or not isinstance(parts, list):
                continue
            cleaned_parts = [clean_sequence(part) for part in parts]
            cleaned_parts = [part for part in cleaned_parts if part]
            if cleaned_parts:
                sequences[taxid] = "N".join(cleaned_parts)
    return sequences


def collect_virobench_rows(
    raw_root: Path,
    tasks: Iterable[str],
) -> tuple[List[ManifestRow], List[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    tasks = list(tasks)
    total = len(tasks)
    virobench_root = raw_root / "virobench"
    for task_idx, task_name in enumerate(tasks, start=1):
        if task_name not in VIROBENCH_TASK_SPECS:
            raise ValueError(f"Unknown ViroBench task: {task_name}")
        spec = VIROBENCH_TASK_SPECS[task_name]
        label_col = spec["label_col"]
        seen_split = False
        for split in ("train", "val", "test"):
            csv_path = virobench_root / virobench_rel_path(task_name, split, ".csv")
            seq_path = virobench_root / virobench_rel_path(task_name, split, ".jsonl")
            if not csv_path.exists():
                missing.append(str(csv_path))
                print(f"[prepare_benchmarks] ViroBench build {task_name} missing metadata: {split}")
                continue
            if not seq_path.exists():
                missing.append(str(seq_path))
                print(f"[prepare_benchmarks] ViroBench build {task_name} missing sequences: {split}")
                continue
            seen_split = True
            df = pd.read_csv(csv_path)
            if "taxid" not in df.columns or label_col not in df.columns:
                raise ValueError(
                    f"ViroBench file missing taxid/{label_col}: {csv_path} columns={list(df.columns)}"
                )
            sequence_by_taxid = read_virobench_sequences(seq_path)
            loaded = 0
            skipped = 0
            for idx, record in df.iterrows():
                taxid = str(record["taxid"]).strip()
                seq = sequence_by_taxid.get(taxid, "")
                if not seq:
                    skipped += 1
                    continue
                label = normalize_label(record[label_col])
                if label is None:
                    skipped += 1
                    continue
                rows.append(
                    ManifestRow(
                        benchmark="virobench",
                        task=task_name,
                        split=split,
                        sequence=seq,
                        label=label,
                        family=str(record.get("family", "")),
                        group="viral_retain",
                        row_id=f"virobench|{task_name}|{split}|{taxid}|{idx}",
                    )
                )
                loaded += 1
            print(
                f"[prepare_benchmarks] ViroBench build {task_idx}/{total} loaded {loaded} rows "
                f"from {csv_path.relative_to(raw_root)} skipped={skipped} (total={len(rows)})"
            )
        if not seen_split:
            task_dir = virobench_root / VIROBENCH_DATASET_DIR / spec["na_type"] / spec["task_kind"] / spec["split_mode"]
            missing.append(str(task_dir))
            print(f"[prepare_benchmarks] ViroBench build {task_idx}/{total} no usable splits: {task_dir}")
    return rows, missing


def read_task_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def find_split_table(task_dir: Path, split: str) -> Optional[Path]:
    candidates = [split]
    if split == "val":
        candidates.extend(["dev", "valid", "validation"])
    for stem in candidates:
        for suffix in (".csv", ".tsv", ".jsonl", ".json"):
            path = task_dir / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def iter_root_task_tables(root: Path) -> Iterable[Path]:
    for suffix in (".csv", ".tsv", ".jsonl", ".json"):
        yield from sorted(root.glob(f"*{suffix}"))


def append_viral_retain_df_rows(
    df: pd.DataFrame,
    *,
    source_path: Path,
    rows: List[ManifestRow],
    allowed_tasks: set[str],
    default_task: Optional[str] = None,
    default_split: Optional[str] = None,
) -> tuple[int, int, set[str], set[str]]:
    label_col = infer_label_column(df)
    seq_series = build_sequence_series(df)
    task_col = infer_task_column(df)
    split_col = infer_split_column(df)
    loaded = 0
    skipped = 0
    seen_tasks: set[str] = set()
    seen_splits: set[str] = set()

    for idx, record in df.iterrows():
        task = normalize_task_name(record[task_col]) if task_col else default_task
        split = normalize_split(str(record[split_col])) if split_col else default_split
        if not task or not split:
            skipped += 1
            continue
        if allowed_tasks and task not in allowed_tasks:
            skipped += 1
            continue
        seq = clean_sequence(seq_series.iloc[idx])
        if not seq:
            skipped += 1
            continue
        label = normalize_label(record[label_col])
        if label is None:
            skipped += 1
            continue
        row_number = len(rows)
        rows.append(
            ManifestRow(
                benchmark="vgue",
                task=task,
                split=split,
                sequence=seq,
                label=label,
                family="",
                group="viral_retain",
                row_id=f"vgue|{task}|{split}|{row_number}",
            )
        )
        loaded += 1
        seen_tasks.add(task)
        seen_splits.add(split)
    return loaded, skipped, seen_tasks, seen_splits


def collect_viral_retain_root_tables(
    viral_retain_root: Path,
    allowed_tasks: set[str],
) -> tuple[List[ManifestRow], List[str], set[str], set[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    present_tasks: set[str] = set()
    complete_tasks: set[str] = set()

    for table_path in iter_root_task_tables(viral_retain_root):
        df = read_task_table(table_path)
        task_col = infer_task_column(df)
        split_col = infer_split_column(df)
        if task_col is None or split_col is None:
            print(
                f"[prepare_benchmarks] viral retain root table skipped "
                f"(needs task+split columns): {table_path}"
            )
            continue
        loaded, skipped, seen_tasks, _seen_splits = append_viral_retain_df_rows(
            df,
            source_path=table_path,
            rows=rows,
            allowed_tasks=allowed_tasks,
        )
        present_tasks.update(seen_tasks)
        print(
            f"[prepare_benchmarks] viral retain root table loaded {loaded} rows "
            f"from {table_path} skipped={skipped} (total={len(rows)})"
        )

    for task in present_tasks:
        splits = {row.split for row in rows if row.task == task}
        if {"train", "val", "test"} <= splits:
            complete_tasks.add(task)
    for task in sorted(allowed_tasks - present_tasks):
        missing.append(str(viral_retain_root / f"{task}.[csv|tsv|jsonl|json]"))
    return rows, missing, present_tasks, complete_tasks


def collect_viral_retain_rows(
    viral_retain_root: Path,
    tasks: Iterable[str],
) -> tuple[List[ManifestRow], List[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    tasks = [normalize_task_name(task) for task in tasks]
    allowed_tasks = set(tasks)
    if not viral_retain_root.exists():
        missing.append(str(viral_retain_root))
        print(f"[prepare_benchmarks] viral retain root missing: {viral_retain_root}")
        return rows, missing

    root_rows, root_missing, root_present_tasks, root_complete_tasks = collect_viral_retain_root_tables(
        viral_retain_root,
        allowed_tasks,
    )
    rows.extend(root_rows)
    if root_present_tasks:
        for task in sorted(root_present_tasks):
            for split in ("train", "val", "test"):
                split_count = sum(1 for row in root_rows if row.task == task and row.split == split)
                if split_count == 0:
                    missing.append(str(viral_retain_root / f"{task}.{split}"))
        missing.extend(
            path
            for path in root_missing
            if normalize_task_name(Path(path).name.split(".", 1)[0]) not in root_present_tasks
        )

    for task_idx, task in enumerate(tasks, start=1):
        if task in root_complete_tasks:
            print(f"[prepare_benchmarks] viral retain build {task_idx}/{len(tasks)} root table covers: {task}")
            continue
        task_dir = viral_retain_root / task
        if not task_dir.exists():
            missing.append(str(task_dir))
            print(f"[prepare_benchmarks] viral retain build {task_idx}/{len(tasks)} missing dir: {task_dir}")
            continue
        seen_split = False
        for split in ("train", "val", "test"):
            table_path = find_split_table(task_dir, split)
            if table_path is None:
                missing.append(str(task_dir / f"{split}.[csv|tsv|jsonl|json]"))
                print(f"[prepare_benchmarks] viral retain build {task} missing split: {split}")
                continue
            seen_split = True
            df = read_task_table(table_path)
            loaded, skipped, _seen_tasks, _seen_splits = append_viral_retain_df_rows(
                df,
                source_path=table_path,
                rows=rows,
                allowed_tasks=allowed_tasks,
                default_task=task,
                default_split=split,
            )
            print(
                f"[prepare_benchmarks] viral retain build {task_idx}/{len(tasks)} "
                f"loaded {loaded} rows from {table_path} "
                f"skipped={skipped} (total={len(rows)})"
            )
        if not seen_split:
            missing.append(str(task_dir))
            print(f"[prepare_benchmarks] viral retain build {task_idx}/{len(tasks)} no usable splits: {task_dir}")
    return rows, missing


def collect_viral_retain_roots(
    viral_retain_roots: Iterable[Path],
    tasks: Iterable[str],
) -> tuple[List[ManifestRow], List[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    roots = list(viral_retain_roots)
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        for root in roots:
            missing.append(str(root))
            print(f"[prepare_benchmarks] viral retain root missing: {root}")
        return rows, missing

    for root in existing_roots:
        root_rows, root_missing = collect_viral_retain_rows(root, tasks)
        rows.extend(root_rows)
        missing.extend(root_missing)
    loaded_tasks = {row.task for row in rows}
    missing = [
        path
        for path in missing
        if normalize_task_name(Path(path).name.split(".", 1)[0]) not in loaded_tasks
    ]
    return rows, missing


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    ensure_parent(path)
    rows = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark", "task", "split", "sequence", "label", "family", "group", "id"],
        )
        writer.writeheader()
        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "benchmark": row.benchmark,
                    "task": row.task,
                    "split": row.split,
                    "sequence": row.sequence,
                    "label": row.label,
                    "family": row.family,
                    "group": row.group,
                    "id": row.row_id,
                }
            )
            if idx == total or idx % 50000 == 0:
                print(f"[prepare_benchmarks] manifest write progress: {idx}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/benchmarks/raw")
    parser.add_argument("--out-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--download-hvue", action="store_true")
    parser.add_argument("--download-gue", action="store_true")
    parser.add_argument("--download-virobench", action="store_true")
    parser.add_argument(
        "--viral-retain-root",
        default=None,
        help="Optional root with task-ready viral retain tables: <root>/<task>/<split>.csv|tsv|jsonl|json",
    )
    parser.add_argument(
        "--viral-retain-tasks",
        nargs="*",
        default=DEFAULT_VIRAL_RETAIN_TASKS,
        help="Viral retain tasks to import when --viral-retain-root exists",
    )
    parser.add_argument(
        "--virobench-tasks",
        nargs="*",
        default=DEFAULT_VIROBENCH_RETAIN_TASKS,
        choices=sorted(VIROBENCH_TASK_SPECS),
        help="ViroBench CLS-Lite tasks to import as viral_retain",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    viral_retain_roots = (
        [Path(args.viral_retain_root)]
        if args.viral_retain_root
        else [raw_root / "viral_retain", raw_root / "vgue"]
    )
    if args.download_hvue:
        downloaded = download_hvue(raw_root)
        print(f"[prepare_benchmarks] downloaded {len(downloaded)} HVUE files")
    if args.download_gue:
        downloaded = download_gue(raw_root)
        print(f"[prepare_benchmarks] downloaded {len(downloaded)} GUE files")
    if args.download_virobench:
        downloaded = download_virobench(raw_root, args.virobench_tasks)
        print(f"[prepare_benchmarks] downloaded {len(downloaded)} ViroBench files")

    hvue_rows, hvue_missing = collect_hvue_rows(raw_root)
    gue_rows, gue_missing = collect_gue_rows(raw_root)
    viral_retain_rows, viral_retain_missing = collect_viral_retain_roots(
        viral_retain_roots,
        args.viral_retain_tasks,
    )
    virobench_rows, virobench_missing = collect_virobench_rows(raw_root, args.virobench_tasks)
    all_rows = hvue_rows + gue_rows + viral_retain_rows + virobench_rows
    write_manifest(Path(args.out_manifest), all_rows)

    print(f"[prepare_benchmarks] wrote manifest: {args.out_manifest}")
    print(
        f"[prepare_benchmarks] rows: hvue={len(hvue_rows)} gue={len(gue_rows)} "
        f"viral_retain={len(viral_retain_rows)} virobench={len(virobench_rows)} total={len(all_rows)}"
    )
    if hvue_missing:
        print(f"[prepare_benchmarks] missing HVUE files: {len(hvue_missing)}")
        for path in hvue_missing[:10]:
            print(f"  - {path}")
    if gue_missing:
        print(f"[prepare_benchmarks] missing GUE task dirs/files: {len(gue_missing)}")
        for path in gue_missing[:10]:
            print(f"  - {path}")
    if viral_retain_missing:
        print(f"[prepare_benchmarks] missing viral retain task dirs/files: {len(viral_retain_missing)}")
        for path in viral_retain_missing[:10]:
            print(f"  - {path}")
    if virobench_missing:
        print(f"[prepare_benchmarks] missing ViroBench files: {len(virobench_missing)}")
        for path in virobench_missing[:10]:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
