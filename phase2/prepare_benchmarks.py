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
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from huggingface_hub import hf_hub_download


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
    "host_range_prediction",
    "dna_vs_rna_virus",
    "hiv1_vs_hiv2",
]

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
    key = name.lower()
    return SPLIT_MAP.get(key, key)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return "".join(ch for ch in seq if ch in {"A", "C", "G", "T", "N"})


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


def collect_hvue_rows(raw_root: Path) -> tuple[List[ManifestRow], List[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    total = len(HVUE_FILES)
    for idx, (rel_path, (task, family, group)) in enumerate(sorted(HVUE_FILES.items()), start=1):
        csv_path = raw_root / "hvue" / rel_path
        if not csv_path.exists():
            missing.append(str(csv_path))
            print(f"[prepare_benchmarks] HVUE build {idx}/{total} missing: {rel_path}")
            continue
        split = normalize_split(csv_path.stem)
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError(f"HVUE file missing required columns: {csv_path}")
        before = len(rows)
        skipped = 0
        for idx, record in df.iterrows():
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
                    row_id=f"hvue|{task}|{split}|{idx}",
                )
            )
        print(
            f"[prepare_benchmarks] HVUE build {idx}/{total} loaded {len(rows) - before} rows "
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


def collect_viral_retain_rows(
    viral_retain_root: Path,
    tasks: Iterable[str],
) -> tuple[List[ManifestRow], List[str]]:
    rows: List[ManifestRow] = []
    missing: List[str] = []
    tasks = list(tasks)
    if not viral_retain_root.exists():
        missing.append(str(viral_retain_root))
        print(f"[prepare_benchmarks] viral retain root missing: {viral_retain_root}")
        return rows, missing

    for task_idx, task in enumerate(tasks, start=1):
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
            label_col = infer_label_column(df)
            seq_series = build_sequence_series(df)
            before = len(rows)
            skipped = 0
            for idx, record in df.iterrows():
                seq = clean_sequence(seq_series.iloc[idx])
                if not seq:
                    skipped += 1
                    continue
                label = normalize_label(record[label_col])
                if label is None:
                    skipped += 1
                    continue
                rows.append(
                    ManifestRow(
                        benchmark="viral_retain",
                        task=task,
                        split=split,
                        sequence=seq,
                        label=label,
                        family="",
                        group="viral_retain",
                        row_id=f"viral_retain|{task}|{split}|{idx}",
                    )
                )
            print(
                f"[prepare_benchmarks] viral retain build {task_idx}/{len(tasks)} "
                f"loaded {len(rows) - before} rows from {table_path} "
                f"skipped={skipped} (total={len(rows)})"
            )
        if not seen_split:
            missing.append(str(task_dir))
            print(f"[prepare_benchmarks] viral retain build {task_idx}/{len(tasks)} no usable splits: {task_dir}")
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
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    viral_retain_root = Path(args.viral_retain_root) if args.viral_retain_root else raw_root / "viral_retain"
    if args.download_hvue:
        downloaded = download_hvue(raw_root)
        print(f"[prepare_benchmarks] downloaded {len(downloaded)} HVUE files")
    if args.download_gue:
        downloaded = download_gue(raw_root)
        print(f"[prepare_benchmarks] downloaded {len(downloaded)} GUE files")

    hvue_rows, hvue_missing = collect_hvue_rows(raw_root)
    gue_rows, gue_missing = collect_gue_rows(raw_root)
    viral_retain_rows, viral_retain_missing = collect_viral_retain_rows(
        viral_retain_root,
        args.viral_retain_tasks,
    )
    all_rows = hvue_rows + gue_rows + viral_retain_rows
    write_manifest(Path(args.out_manifest), all_rows)

    print(f"[prepare_benchmarks] wrote manifest: {args.out_manifest}")
    print(
        f"[prepare_benchmarks] rows: hvue={len(hvue_rows)} gue={len(gue_rows)} "
        f"viral_retain={len(viral_retain_rows)} total={len(all_rows)}"
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


if __name__ == "__main__":
    main()
