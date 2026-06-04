#!/usr/bin/env bash
# Phase 3 — LR-grid recovery attack sweep.
#
# For each unlearned checkpoint, runs SFT and LoRA attacks across a LR grid,
# then selects the LR that maximises recovery AUROC (layers 3-9 mean).
# This mirrors a motivated adversary who would tune the LR to maximise recovery.
#
# Usage:
#   bash phase3/run_attacks.sh <ckpt_root>    # e.g. data/phase2/checkpoints_tuned
#   bash phase3/run_attacks.sh summary        # print best-LR table for all results
#   bash phase3/run_attacks.sh summary <out_root>

set -euo pipefail

# Activate the correct conda environment (requires stripedhyena / Evo dependencies)
# Temporarily disable -u: conda's own activate scripts reference unbound variables
set +u
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate UT-p1
set -u

LR_GRID="1e-5 5e-5 1e-4 5e-4"
STEPS=${STEPS:-2000}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}
DEVICE=${DEVICE:-cuda:0}
OUT_ROOT=${OUT_ROOT:-data/phase3}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

mean_auroc_3_9() {
    # Compute mean AUROC for layers 3-9 from an auroc.csv file.
    python3 - "$1" <<'PYEOF'
import csv, sys, numpy as np
path = sys.argv[1]
vals = []
with open(path) as f:
    for row in csv.DictReader(f):
        if int(row["layer"]) in range(3, 10):
            vals.append(float(row["auroc_after"]))
print(f"{np.mean(vals):.4f}" if vals else "N/A")
PYEOF
}

print_summary() {
    local root="${1:-$OUT_ROOT}"
    python3 - "$root" <<'PYEOF'
import csv, json, os, sys, numpy as np
from collections import defaultdict

root = sys.argv[1]
# group results by (base_run, attack_type) -> list of (lr, auroc_after_mean)
groups = defaultdict(list)
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    auroc_path = os.path.join(d, "auroc.csv")
    meta_path  = os.path.join(d, "meta.json")
    if not (os.path.isfile(auroc_path) and os.path.isfile(meta_path)):
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    if "lr_grid" not in meta:
        continue  # skip old-format results
    attack = meta["attack"]
    run    = meta["run"]
    lr     = meta["lr"]
    steps  = meta["steps"]
    with open(auroc_path) as f:
        vals_after  = [float(r["auroc_after"])  for r in csv.DictReader(f) if int(r["layer"]) in range(3,10)]
    with open(auroc_path) as f:
        vals_before = [float(r["auroc_before"]) for r in csv.DictReader(f) if int(r["layer"]) in range(3,10)]
    key = (run, attack)
    groups[key].append({
        "lr": lr, "steps": steps,
        "before": np.mean(vals_before),
        "after":  np.mean(vals_after),
    })

if not groups:
    print("No lr_grid results found in", root)
    sys.exit(0)

print(f"\n{'Run':<35} {'Attack':<6} {'Best LR':>9} {'Before':>7} {'After':>7} {'Δ':>7}  Steps")
print("-" * 85)
for (run, attack), entries in sorted(groups.items()):
    best = max(entries, key=lambda x: x["after"])
    delta = best["after"] - best["before"]
    print(f"{run:<35} {attack:<6} {best['lr']:>9} {best['before']:>7.3f} {best['after']:>7.3f} {delta:>+7.3f}  {best['steps']}")
PYEOF
}

run_one_ckpt() {
    local ckpt="$1"
    local run_name
    run_name=$(basename "$(dirname "$ckpt")")

    echo ""
    echo "========================================================"
    echo "ATTACKING  $run_name"
    echo "========================================================"

    for LR in $LR_GRID; do
        local sft_dir="$OUT_ROOT/${run_name}_sft_lr${LR}"
        local lora_dir="$OUT_ROOT/${run_name}_lora_lr${LR}"

        if [ -f "$sft_dir/auroc.csv" ]; then
            echo "[skip] SFT lr=$LR already done"
        else
            echo "--- SFT lr=$LR steps=$STEPS ---"
            python phase3/attack_sft.py \
                --ckpt     "$ckpt" \
                --out-dir  "$OUT_ROOT" \
                --run-name "${run_name}_sft_lr${LR}" \
                --lr       "$LR" \
                --steps    "$STEPS" \
                --batch-size "$BATCH" \
                --max-length "$MAX_LEN" \
                --device   "$DEVICE"
        fi

        if [ -f "$lora_dir/auroc.csv" ]; then
            echo "[skip] LoRA lr=$LR already done"
        else
            echo "--- LoRA lr=$LR steps=$STEPS ---"
            python phase3/attack_lora.py \
                --ckpt     "$ckpt" \
                --out-dir  "$OUT_ROOT" \
                --run-name "${run_name}_lora_lr${LR}" \
                --lr       "$LR" \
                --steps    "$STEPS" \
                --batch-size "$BATCH" \
                --max-length "$MAX_LEN" \
                --device   "$DEVICE"
        fi
    done
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "${1:-}" in
    summary)
        print_summary "${2:-$OUT_ROOT}"
        ;;
    "")
        echo "Usage: bash phase3/run_attacks.sh <ckpt_root> | summary [out_root]"
        exit 1
        ;;
    *)
        CKPT_ROOT="$1"
        for run_dir in "$CKPT_ROOT"/*/; do
            ckpt="$run_dir/weights.safetensors"
            [ -f "$ckpt" ] || continue
            run_one_ckpt "$ckpt"
        done
        echo ""
        echo "All attacks complete."
        print_summary "$OUT_ROOT"
        ;;
esac
