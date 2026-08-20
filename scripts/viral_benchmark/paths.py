"""
Central path/config resolution so every script in this directory is portable.

Override any of these with environment variables before launching. Defaults match the
original development cluster; on a new cluster set at minimum VB_HVUE_DIR, VB_SPLITS_V2,
VB_OUT and VB_MMSEQS.

    export VB_ROOT=/scratch/$USER/viral-bench
    export VB_HVUE_DIR=$VB_ROOT/data/hvue
    export VB_OUT=$VB_ROOT/results
    export VB_MMSEQS=$(which mmseqs)
    export HF_HOME=$VB_ROOT/hf_cache
"""
import os

# --- roots -------------------------------------------------------------------
ROOT      = os.environ.get("VB_ROOT", "/data/nvidia/genomic-capability-unlearning")
OUT       = os.environ.get("VB_OUT", os.path.join(ROOT, "results_viral_bench"))
SCRATCH   = os.environ.get("VB_SCRATCH", os.path.join(OUT, "tmp"))

# --- HVUE ---------------------------------------------------------------------
# {task}_{train,validation,test}.parquet  with columns [sequence, label]
# tasks: Host_Tropism, Pathogenecity, Transmissibility
HVUE_DIR  = os.environ.get("VB_HVUE_DIR", "/home/nvidia/glm-locking/data/hvue")

# Pre-built composition-cluster splits from the original project (OPTIONAL --
# these are the baseline-hostile splits; see HANDOFF.md 3.1. Prefer the
# identity-disjoint splits produced by build_ungated_splits.py).
SPLITS_V2 = os.environ.get(
    "VB_SPLITS_V2",
    "/home/nvidia/glm-locking/experiments/hvue_composition_confound/splits_v2")

# --- Evo (only needed by evo_lora_fixed.py) -----------------------------------
# Requires the glm-locking package on PYTHONPATH: src.utils, src.lora,
# scripts.hvue_lora_finetune
LOCK_ROOT = os.environ.get("VB_LOCK_ROOT", "/home/nvidia/glm-locking")

# --- external tools -----------------------------------------------------------
MMSEQS    = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")

# --- downloaded benchmark data ------------------------------------------------
GUE_DIR   = os.environ.get("VB_GUE_DIR", os.path.join(OUT, "gue_viral"))
VIRO_DIR  = os.environ.get("VB_VIRO_DIR", os.path.join(OUT, "virobench"))


def sub(name):
    """Result subdirectory, created on demand."""
    p = os.path.join(OUT, name)
    os.makedirs(p, exist_ok=True)
    return p
