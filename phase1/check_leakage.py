import argparse
import collections
import os
import sys
from typing import Dict, List

import pandas as pd

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for exact sequence leakage across splits.")
    parser.add_argument("--manifest", default="data/phase1/manifest.csv")
    parser.add_argument("--report-top", type=int, default=10)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    rows: List[Dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "id": record.record_id,
                "split": record.split,
                "label": record.label,
                "sequence": record.sequence,
            }
        )
    df = pd.DataFrame(rows)

    df["seq_hash"] = df["sequence"].map(hash)
    counts = df.groupby("seq_hash")["split"].nunique()
    leaked_hashes = counts[counts > 1].index.tolist()

    print(f"Total records: {len(df)}")
    print(f"Unique sequences: {df['seq_hash'].nunique()}")
    print(f"Potential cross-split duplicates: {len(leaked_hashes)}")

    if leaked_hashes:
        leaked = df[df["seq_hash"].isin(leaked_hashes)].copy()
        leaked_summary = leaked.groupby("seq_hash").agg(
            splits=("split", lambda x: ",".join(sorted(set(x)))),
            labels=("label", lambda x: ",".join(sorted(set(map(str, x))))),
            count=("id", "count"),
        )
        print("Top leaked sequences:")
        print(leaked_summary.sort_values("count", ascending=False).head(args.report_top))


if __name__ == "__main__":
    main()
