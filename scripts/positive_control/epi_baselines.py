"""
k-mer and supervised-CNN baselines for GUE's EPI (enhancer-promoter interaction)
task -- paired-sequence variant of scripts/viral_benchmark/gue_baselines.py.
Binary: does this enhancer (3000bp) interact with this promoter (2000bp)?

k-mer: features computed separately for enhancer and promoter, concatenated
(standard "concatenate anchor features" baseline in this benchmark family --
TargetFinder/SPEID/EPIVAN lineage).
CNN: siamese dilated-CNN trunk (shared weights) applied independently to each
sequence, pooled representations concatenated before the classification head.
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)) + "/../viral_benchmark")
import paths as P
warnings.filterwarnings("ignore")
D = P.GUE_DIR
OUT = P.sub("epi_baselines")
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


class SiameseTrunk(nn.Module):
    """Shared dilated-CNN encoder applied to each branch independently."""
    def __init__(self, ch=128, nb=5, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(5, 16)
        L, cin = [], 16
        for i in range(nb):
            L += [nn.Conv1d(cin, ch, 9, padding=2**i*4, dilation=2**i), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop)]
            cin = ch
        self.conv = nn.Sequential(*L)

    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return torch.cat([h.max(-1).values, h.mean(-1)], -1)


class PairedCNN(nn.Module):
    def __init__(self, ch=128, drop=0.2):
        super().__init__()
        self.trunk = SiameseTrunk(ch=ch, drop=drop)
        self.head = nn.Sequential(nn.Linear(4 * ch, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 1))

    def forward(self, xe, xp):
        return self.head(torch.cat([self.trunk(xe), self.trunk(xp)], -1)).squeeze(-1)


def enc(seqs, L):
    X = np.full((len(seqs), L), 4, np.int64)
    for i, s in enumerate(seqs):
        a = MAPI[np.frombuffer(s[:L].encode(), np.uint8)]; X[i, :len(a)] = a
    return X


def metrics(y, p, prob):
    return dict(macro_f1=round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
                accuracy=round(float(accuracy_score(y, p)), 4),
                mcc=round(float(matthews_corrcoef(y, p)), 4),
                auroc=round(float(roc_auc_score(y, prob)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--enh_len", type=int, default=3000); ap.add_argument("--prom_len", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=40000); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(f"{D}/{a.task}__train.csv"); dv = pd.read_csv(f"{D}/{a.task}__dev.csv"); te = pd.read_csv(f"{D}/{a.task}__test.csv")
    if len(tr) > a.cap: tr = tr.sample(a.cap, random_state=0).reset_index(drop=True)
    print(f"{a.task}: train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)
    res = {"task": a.task, "n_train": len(tr), "n_test": len(te)}

    # ---- k-mer (concatenated enhancer+promoter features) ----
    for ks, nm in [((3, 4, 5), "kmer3-5"), ((3, 4, 5, 6), "kmer3-6")]:
        t0 = time.time()
        def feats(d):
            fe = kmer_feats(d.enhancer.tolist(), ks); fp = kmer_feats(d.promoter.tolist(), ks)
            return np.hstack([fe, fp])
        Xtr, Xdv, Xte = feats(tr), feats(dv), feats(te)
        sc = StandardScaler().fit(Xtr); Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
        best = (-1, None)
        for C in [0.003, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=1200, n_jobs=-1).fit(Xtr, tr.label.values)
            s = f1_score(dv.label.values, clf.predict(Xdv), average="macro", zero_division=0)
            if s > best[0]: best = (s, C, clf)
        prob = best[2].predict_proba(Xte)[:, 1]
        m = metrics(te.label.values, best[2].predict(Xte), prob)
        res[nm] = dict(**m, C=best[1], dev_macro_f1=round(best[0], 4), runtime_s=round(time.time() - t0))
        print(f"  {nm:<9} C={best[1]:<6} test macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} mcc={m['mcc']:.4f} auroc={m['auroc']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- siamese CNN ----
    dev = a.device
    Etr, Edv, Ete = (torch.from_numpy(enc(d.enhancer.tolist(), a.enh_len)) for d in (tr, dv, te))
    Ptr, Pdv, Pte = (torch.from_numpy(enc(d.promoter.tolist(), a.prom_len)) for d in (tr, dv, te))
    ytr = torch.tensor(tr.label.values, dtype=torch.float32); ydv = dv.label.values; yte = te.label.values
    cnn_runs = []
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = PairedCNN().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

        def pred(E, Pm):
            m.eval(); o = []
            with torch.no_grad():
                for i in range(0, len(E), 128):
                    o.append(torch.sigmoid(m(E[i:i+128].to(dev), Pm[i:i+128].to(dev))).cpu().numpy())
            m.train(); return np.concatenate(o)

        best = (-1, 0, None); noimp = 0
        for ep in range(1, a.epochs + 1):
            perm = torch.randperm(len(Etr))
            for i in range(0, len(perm) - 32 + 1, 32):
                b = perm[i:i+32]
                loss = F.binary_cross_entropy_with_logits(m(Etr[b].to(dev), Ptr[b].to(dev)), ytr[b].to(dev))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
            p_dv = pred(Edv, Pdv); f_dv = f1_score(ydv, p_dv > 0.5, average="macro", zero_division=0)
            p_te = pred(Ete, Pte); f_te = f1_score(yte, p_te > 0.5, average="macro", zero_division=0)
            print(f"  cnn s{seed} ep{ep:3d} dev_f1={f_dv:.4f} test_f1={f_te:.4f}", flush=True)
            if f_dv > best[0] + 5e-4:
                best = (f_dv, ep, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}); noimp = 0
            else:
                noimp += 1
                if noimp >= 8: print(f"  early stop ep{ep}", flush=True); break
        m.load_state_dict({k: v.to(dev) for k, v in best[2].items()})
        prob = pred(Ete, Pte)
        mm = metrics(yte, prob > 0.5, prob); mm.update(seed=seed, best_epoch=best[1], dev_macro_f1=round(best[0], 4))
        cnn_runs.append(mm); print(f"  CNN s{seed}: test macro_f1={mm['macro_f1']:.4f} mcc={mm['mcc']:.4f}", flush=True)
    res["cnn"] = dict(runs=cnn_runs, n_params=sum(p.numel() for p in PairedCNN().parameters()),
                      mean_macro_f1=round(float(np.mean([r["macro_f1"] for r in cnn_runs])), 4))
    json.dump(res, open(f"{OUT}/{a.task}__baselines.json", "w"), indent=2)
    print(f"DONE {a.task}: kmer3-6={res['kmer3-6']['macro_f1']:.4f}  CNN={res['cnn']['mean_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
