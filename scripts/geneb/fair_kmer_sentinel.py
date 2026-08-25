"""
GENEB's reference ExampleKmerExtractor + run_GENEB.py's default LogisticRegression(max_iter=1000)
produces a degenerate (majority-class, MCC=0.000) fit on iDHS-EL_DNase_I -- confirmed not a data
bug (features have full variance, no NaNs), but a methodological weakness: no feature
standardization, no C tuning, no class weighting.

Redo the k-mer baseline fairly: same 4-mer features, but StandardScaler + C swept on a dev split
carved from train (never touching test), matching this project's own established k-mer-baseline
convention elsewhere (scripts/viral_benchmark/gue_baselines.py). This is the fair number to
compare our gLMs against, not GENEB's raw reference default.
"""
import csv, json, sys
import numpy as np
sys.path.insert(0, "/scratch/10906/arisk/GENEB/harness")
from extractors.example_kmer import ExampleKmerExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TASKS = json.load(open("/scratch/10906/arisk/GENEB/benchmark/benchmark_spec.json"))["tasks"]
DATA = "/scratch/10906/arisk/GENEB/GENEB_data"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0]


def run_task(task_id):
    rows = list(csv.DictReader(open(f"{DATA}/{task_id}.csv")))
    tr_full = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    y_full = np.array([int(r["label"]) for r in tr_full])
    yte = np.array([int(r["label"]) for r in te])

    ext = ExampleKmerExtractor()
    idx = np.arange(len(tr_full))
    idx_tr, idx_dv = train_test_split(idx, test_size=0.15, stratify=y_full, random_state=42)
    tr_texts = [tr_full[i]["text"] for i in idx_tr]
    dv_texts = [tr_full[i]["text"] for i in idx_dv]
    te_texts = [r["text"] for r in te]

    Xtr = ext.extract_embeddings(tr_texts)
    Xdv = ext.extract_embeddings(dv_texts)
    Xte = ext.extract_embeddings(te_texts)
    ytr, ydv = y_full[idx_tr], y_full[idx_dv]

    sc = StandardScaler().fit(Xtr)
    Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)

    best = (-1, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        s = f1_score(ydv, clf.predict(Xdv), average="macro", zero_division=0)
        if s > best[0]:
            best = (s, C)

    # refit on full train (tr+dv) at the selected C, standardized on full train
    Xfull = ext.extract_embeddings([r["text"] for r in tr_full])
    scf = StandardScaler().fit(Xfull)
    clf = LogisticRegression(C=best[1], max_iter=1000, class_weight="balanced").fit(scf.transform(Xfull), y_full)
    pred = clf.predict(scf.transform(Xte))
    mcc = matthews_corrcoef(yte, pred)
    return dict(task=task_id, C=best[1], dev_f1=round(best[0], 4), mcc=round(float(mcc), 4),
                acc=round(float(accuracy_score(yte, pred)), 4), n_pred_classes=len(np.unique(pred)))


if __name__ == "__main__":
    import sys as _s
    only = set(_s.argv[1:]) or None
    out_path = "/scratch/10906/arisk/GENEB/fair_kmer_sentinel_results.json"
    results = json.load(open(out_path)) if __import__("os").path.exists(out_path) else []
    done = {r["task"] for r in results}
    for t in TASKS:
        if only and t["id"] not in only:
            continue
        if t["id"] in done:
            continue
        r = run_task(t["id"])
        print(f"{r['task']:<70} C={r['C']:<7} mcc={r['mcc']:.4f} acc={r['acc']:.4f}", flush=True)
        results.append(r)
        json.dump(results, open(out_path, "w"), indent=2)  # checkpoint after every task
