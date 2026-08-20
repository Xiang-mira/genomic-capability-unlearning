"""
Probe-guided gradient ascent unlearning.

STUB — do NOT run this until Gate 1 (probe_validity_audit.py) confirms that
  decision.hard_stop == false
  AND action not in {"continue_with_strong_identity_confound_risk",
                      "continue_with_identity_confound_risk"}

If probe_validity_audit.json shows a strong shortcut confound (raw baseline AUROC ≥ 0.90),
probe-guided gradient ascent will optimize a signal that is already destroyed by sequence
statistics, not by the model. The resulting checkpoint would still pass fixed-probe
evaluation (the probe direction is disrupted) but a fresh probe retrained on the modified
model will recover separability from the raw features embedded in the residual stream.

This script intentionally exits with a non-zero code if the validity gate has not been
passed. Run probe_validity_audit.py first and pass its output via --validity-audit.

Algorithm (to be implemented after gate passes):
  1. Load base Evo weights and the logistic-regression probe weights w_l for each
     target layer l ∈ {layer_start, ..., layer_end}.
  2. For each batch of forget sequences:
       - Forward pass → extract hidden states H_l ∈ R^{B × L × D}
       - Compute probe score s_l = σ(mean_pool(H_l) @ w_l + b_l)
       - Forget loss = -mean(log(1 - s_l))   [gradient ascent on forget score]
       - Retain loss = CE(model(retain_batch), retain_batch) [standard LM loss]
       - Loss = alpha_forget * L_forget + alpha_retain * L_retain
  3. Update only parameters in the targeted layers (LoRA or full).
  4. After each checkpoint: call eval_unlearn.py --fresh-probe to confirm
     fresh separability is below threshold before continuing.

Usage (after validity gate passes):
  python phase2/unlearn_probe_guided.py \\
    --validity-audit data/phase2/audits/probe_validity_audit.json \\
    --probe-dir data/phase2/probes \\
    --forget-csv data/phase2/splits/forget.csv \\
    --retain-csv data/phase2/splits/retain.csv \\
    --target-layers 5,6,7,8,9 \\
    --out-dir data/phase2/checkpoints_probe_guided/<run_name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BLOCKING_ACTIONS = {
    "continue_with_strong_identity_confound_risk",
    "continue_with_identity_confound_risk",
}


def _check_validity_gate(audit_path: str) -> None:
    """Exit non-zero if probe_validity_audit.json blocks unlearning."""
    p = Path(audit_path)
    if not p.exists():
        print(
            f"[probe-guided] ERROR: validity audit not found at {audit_path}\n"
            "Run probe_validity_audit.py first and pass its output via --validity-audit.",
            file=sys.stderr,
        )
        sys.exit(1)

    with p.open() as f:
        payload = json.load(f)

    decision = payload.get("decision", {})
    action = decision.get("action", "")
    hard_stop = bool(decision.get("hard_stop", False))

    if hard_stop:
        reasons = decision.get("hard_stop_reasons", [])
        print(
            f"[probe-guided] BLOCKED: probe_validity_audit hard_stop=True\n"
            f"  action:  {action}\n"
            f"  reasons: {reasons}\n"
            "Fix the pipeline issues listed above before running unlearning.",
            file=sys.stderr,
        )
        sys.exit(2)

    if action in _BLOCKING_ACTIONS:
        print(
            f"[probe-guided] BLOCKED: probe_validity_audit action={action!r}\n"
            "  Raw sequence statistics (GC content, k-mer features) already achieve\n"
            "  AUROC ≥ 0.80 on the forget target. Probe-guided gradient ascent would\n"
            "  optimize against the shortcut signal, not the model's learned capability.\n"
            "  Result: readout_disruption, not erasure.\n"
            "  Next step: re-examine the forget/retain split for shortcut leakage,\n"
            "  or target a feature that is not shortcut-confounded.",
            file=sys.stderr,
        )
        sys.exit(3)

    print(f"[probe-guided] validity gate passed: action={action!r} hard_stop={hard_stop}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe-guided gradient ascent unlearning (stub)."
    )
    parser.add_argument(
        "--validity-audit",
        required=True,
        metavar="PATH",
        help="Path to probe_validity_audit.json. Script exits if gate not passed.",
    )
    parser.add_argument("--probe-dir", required=True, metavar="PATH")
    parser.add_argument("--forget-csv", required=True, metavar="PATH")
    parser.add_argument("--retain-csv", required=True, metavar="PATH")
    parser.add_argument("--target-layers", default="5,6,7,8,9")
    parser.add_argument("--out-dir", required=True, metavar="PATH")
    parser.add_argument("--alpha-forget", type=float, default=1.0)
    parser.add_argument("--alpha-retain", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fresh-gate-threshold", type=float, default=0.60)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Gate 1 must pass before any model code loads.
    _check_validity_gate(args.validity_audit)

    # ------------------------------------------------------------------ #
    # Implementation goes here — only after Gate 1 is confirmed to pass   #
    # in a real dataset (not just on the current shortcut-confounded set). #
    # ------------------------------------------------------------------ #
    raise NotImplementedError(
        "probe-guided unlearning body not yet implemented. "
        "Validity gate passed — proceed to implement per VALIDITY.md Gate 4 spec."
    )


if __name__ == "__main__":
    main()
