"""Explicit full fine-tuning entry point for supervised benchmark training.

This wrapper keeps full fine-tuning launch commands distinct from the existing
fresh-LoRA entry point while reusing the same benchmark training/evaluation
implementation and result schema.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2 import eval_benchmarks


def main() -> None:
    if "--training-mode" not in sys.argv:
        sys.argv.extend(["--training-mode", "full_ft"])
    eval_benchmarks.main()


if __name__ == "__main__":
    main()
