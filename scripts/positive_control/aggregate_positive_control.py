"""
Merges the classical k-mer/CNN baseline results (from gue_baselines.py, run by
run_baselines.sh over the splits prepare_splits.py built) with the existing
published-model comparison table at
/scratch/10906/arisk/sweep_ft/competitor_comparison_TEST.md, producing the
"devastating contrast" table for the positive-control writeup: on these
real regulatory-genomics tasks, does baseline vs gLM look like the viral
tasks (no headroom) or not?

NTv3 tasks' *_identity_disjoint split is OUR OWN split, built from
InstaDeepAI/nucleotide_transformer_downstream_tasks_revised -- verified
chromosome-disjoint (0% chromosome overlap, 0% exact-sequence duplicates,
independently re-checked). It is NOT confirmed to be the same test set
competitors.csv's numbers were measured on: that file (source of the
Hyena7M/Cad8M/DB2/GROVER/NTv2/GJ-T/GJ-B columns) has no documented
provenance, and the "_revised" HF dataset's own card implies the original
(non-revised) NT-benchmark release used different splits/negative sampling.
Until that provenance is confirmed, treat every "NT" row in this table as a
POSSIBLE split mismatch (our-split-our-number vs their-split-their-number),
not a verified apples-to-apples comparison -- despite our own split being
internally disjoint and correctly measured.

GUE tasks' *_official split is the officially released split (disjointness
unverified -- no coordinate metadata). Independently measured leakage: exact-
duplicate matching found up to ~3.9% (splice_reconstructed) train-test overlap;
a stricter MMseqs2 >=90%-identity clustering check (see
reports/mmseqs_leakage_check.csv) found leakage is meaningfully worse than
that for splice_reconstructed specifically -- 21.5% of test sequences cluster
with a train sequence at 90% identity -- while the prom/tf tasks stay low
(1.4-5.6%) under the same stricter check. Treat the "GUE | Splice All" row's
near-zero gap as partly explained by this leakage, not as a genuine tie.
Kept separately labeled for the same reason.
*_random splits are baseline-only robustness checks (no matching gLM number
exists on that split).

Usage: python aggregate_positive_control.py > ../../reports/positive_control_comparison.md
"""
import json, os, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/viral_benchmark")
import paths as P

RESULT_DIR = f"{P.OUT}/gue_baselines"
COMPETITOR_MD = "/scratch/10906/arisk/sweep_ft/competitor_comparison_TEST.md"

# our_task_name -> (competitor table section, competitor row label)
NTV3_MAP = {
    "enhancers": ("NT", "Enhancer"),
    "enhancers_types": ("NT", "Enhancer Type"),
    "H2AFZ": ("NT", "H2AFZ"),
    "H3K27ac": ("NT", "H3K27ac"),
    "H3K27me3": ("NT", "H3K27me3"),
    "H3K36me3": ("NT", "H3K36me3"),
    "H3K4me1": ("NT", "H3K4me1"),
    "H3K4me2": ("NT", "H3K4me2"),
    "H3K4me3": ("NT", "H3K4me3"),
    "H3K9ac": ("NT", "H3K9ac"),
    "H3K9me3": ("NT", "H3K9me3"),
    "H4K20me1": ("NT", "H4K20me1"),
    "promoter_all": ("NT", "Promoter All"),
    "promoter_no_tata": ("NT", "Promoter NoTATA"),
    "promoter_tata": ("NT", "Promoter TATA"),
    "splice_sites_acceptors": ("NT", "Splice Acceptor"),
    "splice_sites_donors": ("NT", "Splice Donor"),
    "splice_sites_all": ("NT", "Splice All"),
}
GUE_MAP = {
    "gue_prom_core_all": ("GUE", "Core Prom. All"),
    "gue_prom_core_notata": ("GUE", "Core Prom. NoTATA"),
    "gue_prom_core_tata": ("GUE", "Core Prom. TATA"),
    "gue_prom_300_all": ("GUE", "Promoter All"),
    "gue_prom_300_notata": ("GUE", "Promoter NoTATA"),
    "gue_prom_300_tata": ("GUE", "Promoter TATA"),
    "gue_splice_reconstructed": ("GUE", "Splice All"),
    "gue_tf_0": ("GUE", "TF Human 1"),
    "gue_tf_1": ("GUE", "TF Human 2"),
    "gue_tf_2": ("GUE", "TF Human 3"),
    "gue_tf_3": ("GUE", "TF Human 4"),
    "gue_tf_4": ("GUE", "TF Human 5"),
}
MODEL_COLS = ["Hyena7M", "Cad8M", "DB2", "GROVER", "NTv2", "GJ-T", "GJ-B", "S(test)", "B(test)"]


def parse_competitor_md(path):
    text = open(path).read()
    tables = {}
    for section, header_pat in [("GUE", r"### GUE .*?\n(.*?)\n\n"), ("NT", r"### NT .*?\n(.*?)\n\n")]:
        m = re.search(header_pat, text, re.S)
        block = m.group(1)
        rows = {}
        for line in block.splitlines()[2:]:  # skip header + separator row
            cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if cells[0] == "MEAN":
                continue
            rows[cells[0]] = dict(zip(MODEL_COLS, [float(x) for x in cells[1:]]))
        tables[section] = rows
    return tables


def best_baseline(res):
    # mean over CNN seeds, not max -- max-of-N vs a single-point competitor
    # number is an optimistic-selection bias (confirmed to flip the sign of
    # the only two "baseline wins" rows in this table; see audit).
    cnn_runs = res["cnn"]["runs"]
    cnn_mcc_mean = sum(r["mcc"] for r in cnn_runs) / len(cnn_runs)
    cnn_mcc_std = (sum((r["mcc"] - cnn_mcc_mean) ** 2 for r in cnn_runs) / len(cnn_runs)) ** 0.5
    best = max(res["kmer3-5"]["mcc"], res["kmer3-6"]["mcc"], cnn_mcc_mean)
    return best, len(cnn_runs), cnn_mcc_std


def to_markdown_table(df):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(":--" if not pd.api.types.is_numeric_dtype(df[c]) else "--:" for c in df.columns) + "|"
    body = "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))
    return "\n".join([header, sep, body])


def load(task_variant):
    p = f"{RESULT_DIR}/{task_variant}__baselines.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    tables = parse_competitor_md(COMPETITOR_MD)
    rows = []
    for task_map, split_label in [(NTV3_MAP, "identity_disjoint"), (GUE_MAP, "official")]:
        for our_task, (section, label) in task_map.items():
            res_disjoint = load(f"{our_task}_{split_label}")
            res_random = load(f"{our_task}_random")
            comp = tables[section].get(label, {})
            if res_disjoint is None or not comp:
                continue
            bb_disjoint, n_seeds, cnn_std = best_baseline(res_disjoint)
            bb_random, _, _ = best_baseline(res_random) if res_random else (None, None, None)
            best_glm_name = max(comp, key=comp.get) if comp else None
            best_glm_val = comp.get(best_glm_name)
            rows.append(dict(
                section=section, task=label, split=split_label,
                baseline_disjoint=round(bb_disjoint, 4),
                n_seeds=n_seeds, cnn_seed_std=round(cnn_std, 4),
                baseline_random=round(bb_random, 4) if bb_random is not None else None,
                best_glm=best_glm_name, best_glm_mcc=best_glm_val,
                gap=round(best_glm_val - bb_disjoint, 4) if best_glm_val is not None else None,
                **{m: comp.get(m) for m in MODEL_COLS},
            ))
    df = pd.DataFrame(rows).sort_values(["section", "gap"], ascending=[True, False])

    print("# Positive-control comparison: classical baselines vs published gLM/BioJEPA numbers\n")
    print("`baseline_disjoint` = max(k-mer3-5, k-mer3-6, mean-over-seeds CNN) MCC on the split")
    print("matching the competitor table's test set (NTv3: official split, verified")
    print("chromosome-disjoint; GUE: official split, disjointness unverified). CNN uses the MEAN")
    print("across seeds, not the best -- max-of-N vs a single-point competitor number is an")
    print("optimistic-selection bias (this flipped the sign of two rows in an earlier version of")
    print("this table). `n_seeds`/`cnn_seed_std` are reported so any n=1 or high-variance cell is")
    print("visible rather than silent. `baseline_random` = same baseline architecture on a")
    print("pooled-and-reshuffled random split (no matching gLM number exists there -- robustness")
    print("check only). `gap` = best published model's MCC minus baseline_disjoint; positive and")
    print("large = clean positive control. Published-side numbers are themselves single point")
    print("estimates (their own reported +/- is not carried through here) -- treat small |gap|")
    print("values as ties, not wins, in either direction.\n")
    cols = ["section", "task", "baseline_disjoint", "n_seeds", "cnn_seed_std", "baseline_random",
            "best_glm", "best_glm_mcc", "gap"] + MODEL_COLS
    print(to_markdown_table(df[cols]))

    print("\n## Summary")
    print(f"- Mean gap (published best - our baseline), GUE: {df[df.section=='GUE'].gap.mean():.4f}")
    print(f"- Mean gap (published best - our baseline), NT:  {df[df.section=='NT'].gap.mean():.4f}")
    n_baseline_competitive = (df.gap < 0.05).sum()
    print(f"- Tasks where baseline is within 0.05 MCC of the best published model (or beats it): {n_baseline_competitive}/{len(df)}")


if __name__ == "__main__":
    main()
