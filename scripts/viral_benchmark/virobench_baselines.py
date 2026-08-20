"""
ViroBench taxonomy: build usable tasks + k-mer / CNN baselines.

IMPORTANT split finding (verified on the downloaded CSVs):
  the "genus" split (ViroBench's G-split) is NOT genus-disjoint --
  82-84% of test genera also appear in train. Only `taxid` is disjoint (0% overlap).
  the "times" split IS a genuine temporal holdout: train 1982-2017, test 2020-2025,
  zero date overlap, 1-2% species overlap.
=> the temporal split is the strict OOD test; G-split is closer to a record-level holdout.

Sequences are full viral genomes: median 43 kb, max 1.4 Mb. The k-mer baseline sees the
WHOLE genome; transformer context limits truncate heavily (NT-v2 ~12 kb, GENA-LM ~2-4 kb,
HyenaDNA 160 kb). Every model's effective context is recorded so this is not hidden.
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")
D = P.VIRO_DIR
OUT = P.sub("virobench")
CODE = np.full(256, 255, np.uint8); MAPI = np.full(256, 4, np.int64)
for i, c in enumerate("ACGT"):
    CODE[ord(c)] = i; CODE[ord(c.lower())] = i; MAPI[ord(c)] = i; MAPI[ord(c.lower())] = i


def load(mod, split, sp):
    meta = pd.read_csv(f"{D}/{mod}_taxon_{split}__{sp}.csv")
    seqs = {}
    with open(f"{D}/{mod}_taxon_{split}__{sp}_seq.jsonl") as f:
        for line in f:
            r = json.loads(line)
            s = r["sequences"]
            seqs[r["taxid"]] = (max(s, key=len) if isinstance(s, list) and s else (s if isinstance(s, str) else ""))
    meta["sequence"] = meta.taxid.map(seqs)
    return meta.dropna(subset=["sequence"]).reset_index(drop=True)


def kmer_feats(seqs, ks, cap=None):
    out = []
    for k in ks:
        V = 4 ** k; X = np.zeros((len(seqs), V), np.float32)
        pw = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        for r, s in enumerate(seqs):
            if cap: s = s[:cap]
            c = CODE[np.frombuffer(s.encode(), np.uint8)]; ok = c != 255
            if ok.sum() < k: continue
            c = c.astype(np.int64)
            if len(c) - k + 1 <= 0: continue
            w = np.lib.stride_tricks.sliding_window_view(c, k)
            v = np.lib.stride_tricks.sliding_window_view(ok, k).all(1)
            idx = (w @ pw)[v]
            if idx.size: np.add.at(X[r], idx, 1.0); X[r] /= idx.size
        out.append(X)
    return np.hstack(out)


class CNN(nn.Module):
    def __init__(self, ncls, ch=128, nb=6, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(5, 16); L, cin = [], 16
        for i in range(nb):
            L += [nn.Conv1d(cin, ch, 9, padding=2**i*4, dilation=2**i), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop)]
            cin = ch
        self.conv = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Linear(2*ch, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))


def enc(seqs, L):
    X = np.full((len(seqs), L), 4, np.int64)
    for i, s in enumerate(seqs):
        a = MAPI[np.frombuffer(s[:L].encode(), np.uint8)]; X[i, :len(a)] = a
    return X


def met(y, p):
    return dict(macro_f1=round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
                accuracy=round(float(accuracy_score(y, p)), 4),
                mcc=round(float(matthews_corrcoef(y, p)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default="DNA"); ap.add_argument("--split", default="times")
    ap.add_argument("--level", default="family"); ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--kmer_cap", type=int, default=0)   # 0 = whole genome
    ap.add_argument("--cnn_len", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tr, dv, te = (load(a.mod, a.split, s) for s in ["train", "val", "test"])
    lv = a.level
    keep = tr[lv].value_counts(); keep = set(keep[keep >= a.min_count].index) & set(te[lv].dropna())
    tr, dv, te = (d[d[lv].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
    cls = sorted(keep); ix = {c: i for i, c in enumerate(cls)}
    for d in (tr, dv, te): d["y"] = d[lv].map(ix)
    print(f"{a.mod}/taxon/{a.split} level={lv}: classes={len(cls)} train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)
    print(f"  test {lv} seen in train: {len(set(te[lv])&set(tr[lv]))}/{te[lv].nunique()}"
          f" | test genus seen in train: {len(set(te.genus)&set(tr.genus))}/{te.genus.nunique()}", flush=True)
    res = dict(mod=a.mod, split=a.split, level=lv, n_classes=len(cls), n_train=len(tr), n_test=len(te),
               test_level_overlap=len(set(te[lv])&set(tr[lv])), test_genus_overlap=len(set(te.genus)&set(tr.genus)),
               test_genus_n=int(te.genus.nunique()),
               median_seqlen=int(tr.sequence.str.len().median()))
    # ---- k-mer ----
    cap = a.kmer_cap or None
    for ks, nm in [((3,4,5), "kmer3-5"), ((3,4,5,6), "kmer3-6")]:
        t0 = time.time()
        Xtr, Xdv, Xte = (kmer_feats(d.sequence.tolist(), ks, cap) for d in (tr, dv, te))
        sc = StandardScaler().fit(Xtr); Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
        best = (-1, None, None)
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=1500, n_jobs=-1).fit(Xtr, tr.y)
            s = f1_score(dv.y, clf.predict(Xdv), average="macro", zero_division=0)
            if s > best[0]: best = (s, C, clf)
        m = met(te.y, best[2].predict(Xte))
        res[nm] = dict(**m, C=best[1], dev_macro_f1=round(best[0], 4),
                       context="whole_genome" if not cap else f"{cap}bp")
        print(f"  {nm:<9} C={best[1]:<6} ctx={'whole' if not cap else cap} test macroF1={m['macro_f1']:.4f} "
              f"acc={m['accuracy']:.4f} mcc={m['mcc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # ---- CNN ----
    Itr, Idv, Ite = (torch.from_numpy(enc(d.sequence.tolist(), a.cnn_len)) for d in (tr, dv, te))
    ytr = torch.tensor(tr.y.values); runs = []
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = CNN(len(cls)).to(a.device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        def pred(I):
            m.eval(); o = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(I), 32): o.append(m(I[i:i+32].to(a.device)).float().argmax(-1).cpu().numpy())
            m.train(); return np.concatenate(o)
        best = (-1, 0, None); noimp = 0
        for ep in range(1, a.epochs+1):
            perm = torch.randperm(len(Itr))
            for i in range(0, len(perm)-16+1, 16):
                b = perm[i:i+16]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(Itr[b].to(a.device)), ytr[b].to(a.device))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
            f_dv = f1_score(dv.y, pred(Idv), average="macro", zero_division=0)
            f_te = f1_score(te.y, pred(Ite), average="macro", zero_division=0)
            print(f"  cnn s{seed} ep{ep:3d} dev={f_dv:.4f} test={f_te:.4f}", flush=True)
            if f_dv > best[0]+5e-4: best = (f_dv, ep, pred(Ite).copy()); noimp = 0
            else:
                noimp += 1
                if noimp >= 8: break
        mm = met(te.y, best[2]); mm.update(seed=seed, best_epoch=best[1]); runs.append(mm)
        print(f"  CNN s{seed}: test macroF1={mm['macro_f1']:.4f}", flush=True)
    res["cnn"] = dict(runs=runs, context=f"{a.cnn_len}bp",
                      mean_macro_f1=round(float(np.mean([r['macro_f1'] for r in runs])), 4),
                      mean_mcc=round(float(np.mean([r['mcc'] for r in runs])), 4))
    json.dump(res, open(f"{OUT}/{a.mod}_{a.split}_{lv}__baselines.json", "w"), indent=2)
    print(f"DONE {a.mod}/{a.split}/{lv}: kmer3-6={res['kmer3-6']['macro_f1']:.4f} CNN={res['cnn']['mean_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
