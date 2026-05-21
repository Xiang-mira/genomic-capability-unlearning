"""
Build forget / retain splits for Phase 2 unlearning.

Reuses data/host_tropism/manifest.csv:
  - label=1 (human-tropic viral)  -> forget
  - label=0 (non-human-tropic)    -> retain

Train/val/test are taken from the manifest's existing split column so
unlearning training uses only `split == train` and we hold out val/test for
evaluation (consistent with the Phase 1 probes).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.utils import read_manifest


def write_csv(path: str, records, fieldnames) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r.record_id, "label": r.label, "split": r.split,
                "sequence": r.sequence, "source": r.source, "length": r.length,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--out-dir", default="data/phase2/splits")
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    forget = [r for r in records if r.label == 1]
    retain = [r for r in records if r.label == 0]

    fieldnames = ["id", "label", "split", "sequence", "source", "length"]
    write_csv(os.path.join(args.out_dir, "forget.csv"), forget, fieldnames)
    write_csv(os.path.join(args.out_dir, "retain.csv"), retain, fieldnames)

    # Summary
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        side = "forget" if r.label == 1 else "retain"
        counts[r.split][side] += 1
    print(f"Wrote forget ({len(forget)}) and retain ({len(retain)}) to {args.out_dir}")
    for split in ["train", "val", "test"]:
        print(f"  {split}: forget={counts[split]['forget']}  retain={counts[split]['retain']}")


if __name__ == "__main__":
    main()
