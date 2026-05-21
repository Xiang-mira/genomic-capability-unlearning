"""
Phase 3 results visualization.

Produces:
  1. Heatmap: method × attack matrix (mean AUROC L3-9 after attack)
  2. Line plot: AUROC by layer for key runs including tuned checkpoints
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PHASE3_DIR = "data/phase3"
PHASE2_DIR = "data/phase2/checkpoints"
PHASE2_TUNED_DIR = "data/phase2/checkpoints_tuned"

BASELINE_AUROC = {
    0: 0.870, 1: 0.865, 2: 0.859, 3: 0.854, 4: 0.855,
    5: 0.849, 6: 0.853, 7: 0.859, 8: 0.801, 9: 0.838, 10: 0.812,
}

RUNS = ["gd_full", "gd_localized", "gd_probe", "gd_random", "rmu_full", "rmu_localized", "rmu_random"]
ATTACKS = ["sft", "lora"]

RUN_LABELS = {
    "gd_full": "GD full", "gd_localized": "GD local", "gd_probe": "GD probe(0-10)",
    "gd_random": "GD rand",
    "rmu_full": "RMU full", "rmu_localized": "RMU local", "rmu_random": "RMU rand",
}


def load_auroc(run, attack, base_dir=None):
    if base_dir is None:
        base_dir = PHASE3_DIR
    path = os.path.join(base_dir, f"{run}_{attack}", "auroc.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_auroc(run, tuned=False):
    base = PHASE2_TUNED_DIR if tuned else PHASE2_DIR
    path = os.path.join(base, run, "eval_auroc.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def mean_auroc(df, layers=range(3, 10)):
    sub = df[df["layer"].isin(layers)]
    col = "auroc_after" if "auroc_after" in df.columns else "test_auroc"
    return sub[col].mean()


def main():
    os.makedirs(PHASE3_DIR, exist_ok=True)

    # ── Build summary matrix (original runs) ─────────────────────────────────
    matrix = {}
    for run in RUNS:
        p2 = load_phase2_auroc(run)
        after_unlearn = mean_auroc(p2) if p2 is not None else float("nan")
        row = {"after_unlearn": after_unlearn}
        for attack in ATTACKS:
            df = load_auroc(run, attack)
            row[f"after_{attack}"] = mean_auroc(df) if df is not None else float("nan")
        matrix[run] = row

    baseline_mean = np.mean([BASELINE_AUROC[l] for l in range(3, 10)])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 3 Recovery Attack Results — Evo-1-8k-base\n"
                 "Mean probe AUROC (layers 3–9), higher = more viral capability retained",
                 fontsize=12)

    # ── Panel 1: Heatmap ─────────────────────────────────────────────────────
    ax = axes[0]
    col_labels = ["After\nunlearning", "After\nSFT attack", "After\nLoRA attack"]
    col_keys = ["after_unlearn", "after_sft", "after_lora"]
    data = np.array([[matrix[r][k] for k in col_keys] for r in RUNS])
    row_labels = [RUN_LABELS[r] for r in RUNS]

    vmin, vmax = 0.45, baseline_mean
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f"(a) Full method × attack matrix (baseline={baseline_mean:.3f})", fontsize=10, loc="left")

    for i in range(len(RUNS)):
        for j in range(len(col_keys)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    ax.axhline(3.5, color="white", lw=1.5, ls="--")  # separator after gd_random

    # ── Panel 2: Tuned comparison by layer ───────────────────────────────────
    ax = axes[1]
    layers = list(range(0, 11))
    baseline_vals = [BASELINE_AUROC[l] for l in layers]
    ax.plot(layers, baseline_vals, "k-", lw=2.5, label="Baseline", zorder=10)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.axvspan(2.5, 9.5, alpha=0.07, color="steelblue", label="Causal region (3–9)")

    # GD localized ar=5.0 (tuned)
    p2_gd = load_phase2_auroc("gd_localized_ar5.0", tuned=True)
    sft_gd = load_auroc("gd_localized_ar5.0", "sft", base_dir="data/phase3/tuned")
    lora_gd = load_auroc("gd_localized_ar5.0", "lora", base_dir="data/phase3/tuned")

    if p2_gd is not None:
        ax.plot(p2_gd["layer"], p2_gd["test_auroc"],
                color="tab:orange", ls="--", lw=1.2, alpha=0.7, label="GD local ar=5.0 (unlearned)")
    if sft_gd is not None:
        ax.plot(sft_gd["layer"], sft_gd["auroc_after"],
                color="tab:orange", ls="-", marker="o", ms=4, lw=1.8, label="GD local ar=5.0 + SFT")
    if lora_gd is not None:
        ax.plot(lora_gd["layer"], lora_gd["auroc_after"],
                color="tab:orange", ls=":", marker="s", ms=4, lw=1.8, label="GD local ar=5.0 + LoRA")

    # RMU full (original)
    p2_rmu = load_phase2_auroc("rmu_full")
    sft_rmu = load_auroc("rmu_full", "sft")
    lora_rmu = load_auroc("rmu_full", "lora")

    if p2_rmu is not None:
        ax.plot(p2_rmu["layer"], p2_rmu["test_auroc"],
                color="tab:blue", ls="--", lw=1.2, alpha=0.7, label="RMU full (unlearned)")
    if sft_rmu is not None:
        ax.plot(sft_rmu["layer"], sft_rmu["auroc_after"],
                color="tab:blue", ls="-", marker="o", ms=4, lw=1.8, label="RMU full + SFT")
    if lora_rmu is not None:
        ax.plot(lora_rmu["layer"], lora_rmu["auroc_after"],
                color="tab:blue", ls=":", marker="s", ms=4, lw=1.8, label="RMU full + LoRA")

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Test AUROC", fontsize=10)
    ax.set_title("(b) Tuned comparison: GD local (ar=5.0) vs RMU full\n"
                 "comparable forgetting strength, retain_ppl both ≈ 4",
                 fontsize=10, loc="left")
    ax.set_ylim(0.38, 0.95)
    ax.set_xticks(range(0, 11))
    ax.legend(fontsize=7.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(PHASE3_DIR, "phase3_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # ── Text summary ──────────────────────────────────────────────────────────
    print(f"\n{'Run':<22} {'Unlearned':>10} {'After SFT':>10} {'SFT Δ':>8} {'After LoRA':>11} {'LoRA Δ':>8}")
    print("-" * 73)
    for run in RUNS:
        r = matrix[run]
        u, s, lo = r["after_unlearn"], r["after_sft"], r["after_lora"]
        sd = s - u if not np.isnan(s) else float("nan")
        ld = lo - u if not np.isnan(lo) else float("nan")
        print(f"{RUN_LABELS[run]:<22} {u:>10.3f} {s:>10.3f} {sd:>+8.3f} {lo:>11.3f} {ld:>+8.3f}")

    print()
    print("=== Tuned comparison ===")
    print(f"{'Run':<28} {'Unlearned':>10} {'After SFT':>10} {'SFT Δ':>8} {'After LoRA':>11} {'LoRA Δ':>8}")
    print("-" * 79)
    if p2_gd is not None and sft_gd is not None and lora_gd is not None:
        u = mean_auroc(p2_gd)
        s = mean_auroc(sft_gd)
        lo = mean_auroc(lora_gd)
        print(f"{'GD local ar=5.0 (tuned)':<28} {u:>10.3f} {s:>10.3f} {s-u:>+8.3f} {lo:>11.3f} {lo-u:>+8.3f}")
    if p2_rmu is not None and sft_rmu is not None and lora_rmu is not None:
        u = mean_auroc(p2_rmu)
        s = mean_auroc(sft_rmu)
        lo = mean_auroc(lora_rmu)
        print(f"{'RMU full (orig)':<28} {u:>10.3f} {s:>10.3f} {s-u:>+8.3f} {lo:>11.3f} {lo-u:>+8.3f}")
    print(f"{'Baseline':<28} {baseline_mean:>10.3f}")


if __name__ == "__main__":
    main()

import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PHASE3_DIR = "data/phase3"
PHASE2_DIR = "data/phase2/checkpoints"

BASELINE_AUROC = {
    0: 0.870, 1: 0.865, 2: 0.859, 3: 0.854, 4: 0.855,
    5: 0.849, 6: 0.853, 7: 0.859, 8: 0.801, 9: 0.838, 10: 0.812,
}

RUNS = ["gd_full", "gd_localized", "gd_probe", "gd_random", "rmu_full", "rmu_localized", "rmu_random"]
ATTACKS = ["sft", "lora"]

RUN_LABELS = {
    "gd_full": "GD full", "gd_localized": "GD local", "gd_probe": "GD probe(0-10)",
    "gd_random": "GD rand",
    "rmu_full": "RMU full", "rmu_localized": "RMU local", "rmu_random": "RMU rand",
}


def load_auroc(run, attack):
    path = os.path.join(PHASE3_DIR, f"{run}_{attack}", "auroc.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_auroc(run):
    path = os.path.join(PHASE2_DIR, run, "eval_auroc.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def mean_auroc(df, layers=range(3, 10)):
    sub = df[df["layer"].isin(layers)]
    col = "auroc_after" if "auroc_after" in df.columns else "test_auroc"
    return sub[col].mean()


def main():
    os.makedirs(PHASE3_DIR, exist_ok=True)

    # ── Build summary matrix ──────────────────────────────────────────────────
    # Rows: runs, Cols: [after_unlearn, after_sft, after_lora]
    matrix = {}
    for run in RUNS:
        p2 = load_phase2_auroc(run)
        after_unlearn = mean_auroc(p2) if p2 is not None else float("nan")
        row = {"after_unlearn": after_unlearn}
        for attack in ATTACKS:
            df = load_auroc(run, attack)
            row[f"after_{attack}"] = mean_auroc(df) if df is not None else float("nan")
        matrix[run] = row

    baseline_mean = np.mean([BASELINE_AUROC[l] for l in range(3, 10)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 3 Recovery Attack Results — Evo-1-8k-base\n"
                 "Mean probe AUROC (layers 3–9), higher = more viral capability",
                 fontsize=12)

    # ── Panel 1: Heatmap ─────────────────────────────────────────────────────
    ax = axes[0]
    col_labels = ["After\nunlearning", "After\nSFT attack", "After\nLoRA attack"]
    col_keys = ["after_unlearn", "after_sft", "after_lora"]
    data = np.array([[matrix[r][k] for k in col_keys] for r in RUNS])
    row_labels = [RUN_LABELS[r] for r in RUNS]

    # Normalize to [0,1] relative to baseline
    vmin, vmax = 0.5, baseline_mean
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f"(a) AUROC matrix (baseline={baseline_mean:.3f})", fontsize=10, loc="left")

    for i in range(len(RUNS)):
        for j in range(len(col_keys)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    ax.axhline(2.5, color="white", lw=1.5, ls="--")  # separator GD / RMU

    # ── Panel 2: AUROC recovery by layer for key runs ─────────────────────────
    ax = axes[1]
    layers = list(range(0, 11))
    baseline_vals = [BASELINE_AUROC[l] for l in layers]
    ax.plot(layers, baseline_vals, "k-", lw=2.5, label="Baseline", zorder=10)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.axvspan(2.5, 9.5, alpha=0.07, color="steelblue")

    colors = {"gd_localized": "tab:orange", "rmu_full": "tab:blue"}
    for run, color in colors.items():
        # After unlearn
        p2 = load_phase2_auroc(run)
        if p2 is not None:
            ax.plot(p2["layer"], p2["test_auroc"], color=color, ls="--",
                    lw=1.2, alpha=0.6, label=f"{RUN_LABELS[run]} (unlearned)")
        # After SFT
        df_sft = load_auroc(run, "sft")
        if df_sft is not None:
            ax.plot(df_sft["layer"], df_sft["auroc_after"], color=color, ls="-",
                    marker="o", ms=4, lw=1.8, label=f"{RUN_LABELS[run]} + SFT")
        # After LoRA
        df_lora = load_auroc(run, "lora")
        if df_lora is not None:
            ax.plot(df_lora["layer"], df_lora["auroc_after"], color=color, ls=":",
                    marker="s", ms=4, lw=1.8, label=f"{RUN_LABELS[run]} + LoRA")

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Test AUROC", fontsize=10)
    ax.set_title("(b) AUROC recovery by layer\n(GD local & RMU full only)", fontsize=10, loc="left")
    ax.set_ylim(0.40, 0.95)
    ax.set_xticks(range(0, 11))
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(PHASE3_DIR, "phase3_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # ── Text summary ──────────────────────────────────────────────────────────
    print(f"\n{'Run':<14} {'Unlearned':>10} {'After SFT':>10} {'SFT Δ':>8} {'After LoRA':>11} {'LoRA Δ':>8}")
    print("-" * 65)
    for run in RUNS:
        r = matrix[run]
        u = r["after_unlearn"]
        s = r["after_sft"]
        lo = r["after_lora"]
        sd = s - u if not np.isnan(s) else float("nan")
        ld = lo - u if not np.isnan(lo) else float("nan")
        print(f"{RUN_LABELS[run]:<14} {u:>10.3f} {s:>10.3f} {sd:>+8.3f} {lo:>11.3f} {ld:>+8.3f}")
    print(f"{'Baseline':<14} {baseline_mean:>10.3f}")


if __name__ == "__main__":
    main()
