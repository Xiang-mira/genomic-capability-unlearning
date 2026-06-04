"""
Phase 2 internal diagnostic visualization.

Produces a 2-panel figure:
  Left:  host-tropism probe AUROC by layer for all runs vs Phase 1 baseline
  Right: forget_ppl vs retain_ppl diagnostic scatter
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE = "data/phase2/checkpoints"
BASELINE_AUROC = {
    0: 0.870, 1: 0.865, 2: 0.859, 3: 0.854, 4: 0.855,
    5: 0.849, 6: 0.853, 7: 0.859, 8: 0.801, 9: 0.838, 10: 0.812,
}

RUNS = [
    ("gd_full",       "GD full",       "tab:red",    "-",  "o"),
    ("gd_localized",  "GD localized",  "tab:orange", "-",  "s"),
    ("gd_probe",      "GD probe(0-10)","tab:brown",  "-",  "D"),
    ("gd_random",     "GD random",     "tab:pink",   "--", "^"),
    ("rmu_full",      "RMU full",      "tab:blue",   "-",  "o"),
    ("rmu_localized", "RMU localized", "tab:cyan",   "-",  "s"),
    ("rmu_random",    "RMU random",    "tab:purple", "--", "^"),
]


def load_run(run_name):
    auroc = pd.read_csv(os.path.join(BASE, run_name, "eval_auroc.csv"))
    with open(os.path.join(BASE, run_name, "eval_ppl.json")) as f:
        ppl = json.load(f)
    return auroc, ppl


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 2 Internal Diagnostics — Evo-1-8k-base\nHost Tropism: Human-tropic vs Non-human-tropic",
                 fontsize=12)

    # ── Panel 1: AUROC by layer ───────────────────────────────────────────────
    ax = axes[0]
    layers_base = sorted(BASELINE_AUROC.keys())
    ax.plot(layers_base, [BASELINE_AUROC[l] for l in layers_base],
            "k-", lw=2.5, label="Baseline (Phase 1)", zorder=10)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="Random chance")
    ax.axvspan(2.5, 9.5, alpha=0.07, color="steelblue", label="Localized region (3–9)")

    for run_name, label, color, ls, marker in RUNS:
        try:
            auroc, _ = load_run(run_name)
        except FileNotFoundError:
            continue
        ax.plot(auroc["layer"], auroc["test_auroc"],
                color=color, ls=ls, marker=marker, ms=5, lw=1.5, label=label)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Test AUROC", fontsize=10)
    ax.set_title("(a) Probe AUROC after unlearning", fontsize=10, loc="left")
    ax.set_ylim(0.40, 0.95)
    ax.set_xticks(range(0, 11))
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.25)

    # ── Panel 2: forget_ppl vs retain_ppl trade-off ───────────────────────────
    ax = axes[1]
    baseline_ppl = 4.2  # approximate baseline perplexity
    ax.axvline(baseline_ppl, color="gray", lw=0.8, ls=":", label=f"Baseline ppl ≈ {baseline_ppl}")
    ax.axhline(baseline_ppl, color="gray", lw=0.8, ls=":")

    for run_name, label, color, ls, marker in RUNS:
        try:
            _, ppl = load_run(run_name)
        except FileNotFoundError:
            continue
        fp = ppl["forget_val_perplexity"]
        rp = ppl["retain_val_perplexity"]
        ax.scatter(fp, rp, color=color, marker=marker, s=80, zorder=5, label=label)
        ax.annotate(label, (fp, rp), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5, color=color)

    ax.set_xlabel("Forget perplexity ↑ (more forgetting)", fontsize=10)
    ax.set_ylabel("Retain perplexity ↓ (less collateral damage)", fontsize=10)
    ax.set_title("(b) PPL diagnostics\n(not a benchmark trade-off)", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = "data/phase2/phase2_results.png"
    os.makedirs("data/phase2", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # ── Text summary table ────────────────────────────────────────────────────
    print("\n=== Summary Table ===")
    print(f"{'Run':<16} {'AUROC L3-9 mean':>16} {'AUROC Δ':>9} {'forget_ppl':>11} {'retain_ppl':>11}")
    base_mean = np.mean([BASELINE_AUROC[l] for l in range(3, 10)])
    for run_name, label, *_ in RUNS:
        try:
            auroc, ppl = load_run(run_name)
        except FileNotFoundError:
            continue
        sub = auroc[auroc["layer"].between(3, 9)]
        mean_auroc = sub["test_auroc"].mean()
        delta = mean_auroc - base_mean
        fp = ppl["forget_val_perplexity"]
        rp = ppl["retain_val_perplexity"]
        print(f"{label:<16} {mean_auroc:>16.3f} {delta:>+9.3f} {fp:>11.2f} {rp:>11.2f}")
    print(f"{'Baseline':<16} {base_mean:>16.3f} {'—':>9} {'~4.2':>11} {'~4.2':>11}")
    print("\nNote: gd_probe targets layers 0-10 (probe-based), gd_localized targets layers 3-9 (patching-based).")


if __name__ == "__main__":
    main()
