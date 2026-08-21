
"""
ViroBench-PROTOCOL frozen probe (Phase 1 head-to-head).

ViroBench's own stated method for classification, verbatim from the paper:
  "We extract embedding features from each model to train a standardized, lightweight
   classification head, minimizing biases from heterogeneous tokenizers and architectures."
plus: training samples fixed-length windows (512/1024/2048 bp); at EVAL "all available
windows" are embedded and window-level predictions aggregated to a sequence-level decision;
LRs tuned 1e-2..1e-4 and results reported as the MEAN over those settings.

So this script does NOT fine-tune. It:
  1. tiles every genome into non-overlapping W-bp windows
  2. embeds every window with the FROZEN model (mean-pool over tokens)
  3. fits a logistic-regression head on TRAIN windows
  4. at eval, predicts every TEST window and aggregates by mean predicted probability
  5. selects C on the DEV split only; reports TEST once
  6. saves per-example predictions for paired bootstrap

Window budget is capped per genome (--max_win) so one 1.4 Mb outlier cannot dominate;
the cap is recorded in the output.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import paths as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
try:
    import transformers; transformers.logging.set_verbosity_error()
except Exception: pass

OUT = P.sub("virobench_frozen")
MODELS = {   # name: (hf_id, loader, hidden, bp_per_token)
 "lucavirus":  ("LucaGroup/LucaVirus-default-step3.8M", "AutoModel", 2560, 1),
 "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", "AutoModel", 256, 1),
 "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t", "AutoModel", 768, 6),
 "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "AutoModelForMaskedLM", 1024, 6),
}

def load_split(mod, split, sp):
    meta = pd.read_csv(f"{P.VIRO_DIR}/{mod}_taxon_{split}__{sp}.csv")
    seqs = {}
    with open(f"{P.VIRO_DIR}/{mod}_taxon_{split}__{sp}_seq.jsonl") as f:
        for line in f:
            r = json.loads(line); s = r["sequences"]
            seqs[r["taxid"]] = (max(s, key=len) if isinstance(s, list) and s else (s if isinstance(s, str) else ""))
    meta["sequence"] = meta.taxid.map(seqs)
    return meta.dropna(subset=["sequence"]).reset_index(drop=True)

def windows(seq, W, max_win):
    n = max(1, len(seq) // W)
    n = min(n, max_win)
    return [seq[i*W:(i+1)*W] for i in range(n)]

@torch.no_grad()
def embed(model, tok, texts, W, bp_per_tok, bs, dev):
    ml = W // bp_per_tok + 2
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], return_tensors="pt", padding="max_length",
                truncation=True, max_length=ml)
        ids = e["input_ids"].to(dev)
        am = e.get("attention_mask")
        kw = dict(input_ids=ids)
        if am is not None: kw["attention_mask"] = am.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**kw, output_hidden_states=True)
        h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o.last_hidden_state
        if am is None:
            pooled = h.mean(1)
        else:
            mm = am.to(dev).unsqueeze(-1).to(h.dtype)
            pooled = (h*mm).sum(1) / mm.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.vstack(out)

def build(df, W, max_win):
    X_txt, owner = [], []
    for i, s in enumerate(df.sequence.values):
        for w in windows(s, W, max_win):
            X_txt.append(w); owner.append(i)
    return X_txt, np.array(owner)

def agg_predict(clf, E, owner, n_items, n_cls):
    pr = clf.predict_proba(E)
    acc = np.zeros((n_items, n_cls)); cnt = np.zeros(n_items)
    for k, o in enumerate(owner):
        acc[o] += pr[k]; cnt[o] += 1
    cnt[cnt == 0] = 1
    return (acc / cnt[:, None]).argmax(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--mod", default="ALL"); ap.add_argument("--split", default="times")
    ap.add_argument("--level", default="family"); ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--max_win", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--cap_train", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    mid, how, hid, bpt = MODELS[a.model]
    tag = f"{a.model}__frozen__{a.mod}_{a.split}_{a.level}__W{a.window}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"skip {tag}", flush=True); return
    t0 = time.time()
    tr, dv, te = (load_split(a.mod, a.split, s) for s in ["train", "val", "test"])
    keep = tr[a.level].value_counts()
    keep = set(keep[keep >= a.min_count].index) & set(te[a.level].dropna())
    tr, dv, te = (d[d[a.level].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
    if a.cap_train and len(tr) > a.cap_train:
        tr = tr.sample(a.cap_train, random_state=0).reset_index(drop=True)
    cls = sorted(keep); ix = {c: i for i, c in enumerate(cls)}
    for d in (tr, dv, te): d["y"] = d[a.level].map(ix)
    print(f"{a.model} FROZEN W={a.window}bp maxwin={a.max_win} | {a.mod}/{a.split}/{a.level} "
          f"classes={len(cls)} train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)

    cls_ = {"AutoModel": AutoModel, "AutoModelForMaskedLM": AutoModelForMaskedLM}[how]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = cls_.from_pretrained(mid, trust_remote_code=True).to(a.device).eval()
    npar = sum(p.numel() for p in model.parameters())

    packs = {}
    for nm, d in [("train", tr), ("dev", dv), ("test", te)]:
        txt, own = build(d, a.window, a.max_win)
        t = time.time()
        E = embed(model, tok, txt, a.window, bpt, a.bs, a.device)
        print(f"  {nm}: {len(txt)} windows -> {E.shape} in {time.time()-t:.0f}s "
              f"({len(txt)/max(time.time()-t,1e-9):.1f} win/s)", flush=True)
        packs[nm] = (E, own, d.y.values)
    del model; torch.cuda.empty_cache()

    sc = StandardScaler().fit(packs["train"][0])
    Etr, Edv, Ete = (sc.transform(packs[k][0]) for k in ["train", "dev", "test"])
    ytr_w = packs["train"][2][packs["train"][1]]        # window-level labels
    best = (-1, None, None)
    for C in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=1200, n_jobs=-1).fit(Etr, ytr_w)
        p = agg_predict(clf, Edv, packs["dev"][1], len(dv), len(cls))
        s = f1_score(packs["dev"][2], p, average="macro", zero_division=0)
        print(f"  C={C:<7} dev macro-F1={s:.4f}", flush=True)
        if s > best[0]: best = (s, C, clf)
    clf = best[2]
    pte = agg_predict(clf, Ete, packs["test"][1], len(te), len(cls))
    yte = packs["test"][2]
    res = dict(model=a.model, hf_id=mid, regime="frozen_probe_viroBench_protocol",
               n_params=npar, mod=a.mod, split=a.split, level=a.level,
               window_bp=a.window, max_win_per_genome=a.max_win, n_classes=len(cls),
               n_train=len(tr), n_dev=len(dv), n_test=len(te), C=best[1],
               dev_macro_f1=round(best[0], 4),
               test_macro_f1=round(float(f1_score(yte, pte, average="macro", zero_division=0)), 4),
               test_micro_f1=round(float(f1_score(yte, pte, average="micro", zero_division=0)), 4),
               test_accuracy=round(float(accuracy_score(yte, pte)), 4),
               test_mcc=round(float(matthews_corrcoef(yte, pte)), 4),
               per_class_f1={cls[i]: round(float(v), 4) for i, v in
                             enumerate(f1_score(yte, pte, average=None, labels=range(len(cls)), zero_division=0))},
               runtime_s=round(time.time()-t0))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
    np.savez_compressed(f"{OUT}/{tag}__preds.npz", taxid=te.taxid.values,
                        y_true=yte, y_pred=pte, classes=np.array(cls, dtype=object))
    print(f"  RESULT {tag}: test macro-F1={res['test_macro_f1']:.4f} mcc={res['test_mcc']:.4f} "
          f"acc={res['test_accuracy']:.4f} (C={best[1]}, {res['runtime_s']}s)", flush=True)

if __name__ == "__main__":
    main()
