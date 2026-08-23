"""
k-mer and supervised-CNN baselines on the two GUE viral tasks the repo dropped:
  virus_covid       9 SARS-CoV-2 variants, 73,335/9,166/9,168, ~1kb   (GUE metric: F1)
  virus_species_40  25 virus species,       4,000/500/500,     ~5kb

These are the reference numbers any gLM must beat. Multiclass -> macro-F1 primary,
accuracy and MCC reported alongside. C / LR selected on dev, reported on test.
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
D = P.GUE_DIR
OUT = P.sub("gue_baselines")
CODE = np.full(256, 255, np.uint8); MAPI = np.full(256, 4, np.int64)
for i, c in enumerate("ACGT"):
    CODE[ord(c)] = i; CODE[ord(c.lower())] = i; MAPI[ord(c)] = i; MAPI[ord(c.lower())] = i


def kmer_feats(seqs, ks):
    mats = []
    for k in ks:
        V = 4 ** k; X = np.zeros((len(seqs), V), np.float32)
        pw = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        for r, s in enumerate(seqs):
            c = CODE[np.frombuffer(s.encode(), np.uint8)]; ok = c != 255
            if ok.sum() < k: continue
            c = c.astype(np.int64)
            if len(c) - k + 1 <= 0: continue
            w = np.lib.stride_tricks.sliding_window_view(c, k)
            v = np.lib.stride_tricks.sliding_window_view(ok, k).all(1)
            idx = (w @ pw)[v]
            if idx.size: np.add.at(X[r], idx, 1.0); X[r] /= idx.size
        mats.append(X)
    return np.hstack(mats)


class CNN(nn.Module):
    def __init__(self, ncls, ch=128, nb=5, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(5, 16)
        L, cin = [], 16
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


def metrics(y, p):
    return dict(macro_f1=round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
                accuracy=round(float(accuracy_score(y, p)), 4),
                mcc=round(float(matthews_corrcoef(y, p)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--test_csv", default=None); ap.add_argument("--maxlen", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=40000); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(f"{D}/{a.task}__train.csv"); dv = pd.read_csv(f"{D}/{a.task}__dev.csv")
    te = pd.read_csv(a.test_csv if a.test_csv else f"{D}/{a.task}__test.csv")
    if a.test_csv:
        print(f"  TEST SET OVERRIDE: {a.test_csv} (n={len(te)})", flush=True)
    if len(tr) > a.cap: tr = tr.sample(a.cap, random_state=0).reset_index(drop=True)
    ncls = int(pd.concat([tr, dv, te]).label.nunique())
    print(f"{a.task}: train={len(tr)} dev={len(dv)} test={len(te)} classes={ncls}", flush=True)
    res = {"task": a.task, "n_classes": ncls, "n_train": len(tr), "n_test": len(te)}

    # ---- k-mer ----
    for ks, nm in [((3,4,5), "kmer3-5"), ((3,4,5,6), "kmer3-6")]:
        t0 = time.time()
        Xtr, Xdv, Xte = (kmer_feats(d.sequence.tolist(), ks) for d in (tr, dv, te))
        sc = StandardScaler().fit(Xtr); Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
        best = (-1, None)
        for C in [0.003, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=1200, n_jobs=-1).fit(Xtr, tr.label.values)
            s = f1_score(dv.label.values, clf.predict(Xdv), average="macro", zero_division=0)
            if s > best[0]: best = (s, C, clf)
        m = metrics(te.label.values, best[2].predict(Xte))
        res[nm] = dict(**m, C=best[1], dev_macro_f1=round(best[0], 4), runtime_s=round(time.time()-t0))
        print(f"  {nm:<9} C={best[1]:<6} test macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} mcc={m['mcc']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- CNN ----
    dev = a.device
    Itr, Idv, Ite = (torch.from_numpy(enc(d.sequence.tolist(), a.maxlen)) for d in (tr, dv, te))
    ytr = torch.tensor(tr.label.values); ydv = dv.label.values; yte = te.label.values
    cnn_runs = []
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = CNN(ncls).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        def pred(I):
            m.eval(); o = []
            with torch.no_grad():
                for i in range(0, len(I), 256): o.append(m(I[i:i+256].to(dev)).argmax(-1).cpu().numpy())
            m.train(); return np.concatenate(o)
        best = (-1, 0, None); noimp = 0
        for ep in range(1, a.epochs + 1):
            perm = torch.randperm(len(Itr))
            for i in range(0, len(perm) - 64 + 1, 64):
                b = perm[i:i+64]
                loss = F.cross_entropy(m(Itr[b].to(dev)), ytr[b].to(dev))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
            f_dv = f1_score(ydv, pred(Idv), average="macro", zero_division=0)
            f_te = f1_score(yte, pred(Ite), average="macro", zero_division=0)
            print(f"  cnn s{seed} ep{ep:3d} dev_f1={f_dv:.4f} test_f1={f_te:.4f}", flush=True)
            if f_dv > best[0] + 5e-4:
                best = (f_dv, ep, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}); noimp = 0
            else:
                noimp += 1
                if noimp >= 8: print(f"  early stop ep{ep}", flush=True); break
        m.load_state_dict({k: v.to(dev) for k, v in best[2].items()})
        mm = metrics(yte, pred(Ite)); mm.update(seed=seed, best_epoch=best[1], dev_macro_f1=round(best[0], 4))
        cnn_runs.append(mm); print(f"  CNN s{seed}: test macro_f1={mm['macro_f1']:.4f}", flush=True)
    res["cnn"] = dict(runs=cnn_runs, n_params=sum(p.numel() for p in CNN(ncls).parameters()),
                      mean_macro_f1=round(float(np.mean([r["macro_f1"] for r in cnn_runs])), 4))
    json.dump(res, open(f"{OUT}/{a.task}__baselines" + ("__dedup" if a.test_csv else "") + f".json", "w"), indent=2)
    print(f"DONE {a.task}: kmer3-6={res['kmer3-6']['macro_f1']:.4f}  CNN={res['cnn']['mean_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
