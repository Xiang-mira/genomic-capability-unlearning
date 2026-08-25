"""
Supervised ab-initio CNN baseline on the EXACT v2 composition-confound splits.

DART-Eval's central finding is that DNA language models lose to supervised CNNs, not
to k-mer. No CNN baseline exists in either project. This adds one, on the same
train/val partitions the LoRA numbers were computed on, with the same discipline:
early stopping on a train-internal slice, AUROC reported on the held-out partition.

Reference numbers to beat (Host_Tropism / cluster_disjoint):
    kmer3-6 = 0.8034      Evo LoRA = 0.8173 (+0.0139)
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")

SPL = P.SPLITS_V2
OUT = P.sub("hvue_cnn")
MAP = np.full(256, 4, np.int64)
for i, c in enumerate("ACGT"):
    MAP[ord(c)] = i; MAP[ord(c.lower())] = i


def encode(seqs, L=1000):
    X = np.full((len(seqs), L), 4, np.int64)
    for i, s in enumerate(seqs):
        a = MAP[np.frombuffer(s[:L].encode(), np.uint8)]
        X[i, :len(a)] = a
    return X


class DilatedCNN(nn.Module):
    """~1-2M params, 5 dilated conv blocks, global max+mean pool."""
    def __init__(self, ch=128, n_blocks=5, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(5, 16)
        layers, cin = [], 16
        for i in range(n_blocks):
            layers += [nn.Conv1d(cin, ch, 9, padding=2 ** i * 4, dilation=2 ** i),
                       nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop)]
            cin = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * ch, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1)).squeeze(-1)


def run(task, split, seed, lr, epochs, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    df = pd.read_parquet(f"{SPL}/{task}__{split}.parquet")
    tr_all = df[df.partition == "train"].reset_index(drop=True)
    ev = df[df.partition == "val"].reset_index(drop=True)
    rng = np.random.default_rng(seed); iv = np.zeros(len(tr_all), bool)
    for c in tr_all.label.unique():
        idx = np.where(tr_all.label.values == c)[0]; rng.shuffle(idx)
        iv[idx[:int(0.15 * len(idx))]] = True
    tr, ivd = tr_all[~iv].reset_index(drop=True), tr_all[iv].reset_index(drop=True)

    Xtr = torch.from_numpy(encode(tr.sequence.tolist())); ytr = torch.tensor(tr.label.values, dtype=torch.float32)
    Xiv = torch.from_numpy(encode(ivd.sequence.tolist())); yiv = ivd.label.values.astype(int)
    Xev = torch.from_numpy(encode(ev.sequence.tolist())); yev = ev.label.values.astype(int)

    m = DilatedCNN().to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def score(X):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                out.append(m(X[i:i + 512].to(dev)).float().cpu().numpy())
        m.train(); return np.concatenate(out)

    best = dict(iv=0.0, ep=0, state=None); traj = []
    BS = 128
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm) - BS + 1, BS):
            b = perm[i:i + BS]
            loss = F.binary_cross_entropy_with_logits(m(Xtr[b].to(dev)), ytr[b].to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
        a_iv = roc_auc_score(yiv, score(Xiv)); a_ev = roc_auc_score(yev, score(Xev))
        traj.append(dict(epoch=ep, iv_auroc=round(a_iv, 4), ev_auroc=round(a_ev, 4)))
        print(f"  [{task}/{split}/s{seed}/lr{lr:g}] ep{ep:3d} iv={a_iv:.4f} ev={a_ev:.4f}", flush=True)
        if a_iv > best["iv"] + 5e-4:
            best.update(iv=a_iv, ep=ep, state={k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
            noimp = 0
        else:
            noimp = noimp + 1 if ep > 1 else 0
            if noimp >= 8:
                print(f"  early stop @ ep{ep}", flush=True); break
    m.load_state_dict({k: v.to(dev) for k, v in best["state"].items()})
    s_iv, s_ev = score(Xiv), score(Xev)
    fpr, tpr, th = roc_curve(yiv, s_iv); thr = float(th[np.argmax(tpr - fpr)])
    res = dict(task=task, split=split, seed=seed, lr=lr, model="dilated_cnn", n_params=n_par,
               best_epoch=best["ep"], iv_auroc=round(float(best["iv"]), 4),
               ev_auroc=round(float(roc_auc_score(yev, s_ev)), 4),
               ev_mcc=round(float(matthews_corrcoef(yev, (s_ev >= thr).astype(int))), 4),
               n_train=len(tr), n_iv=len(ivd), n_ev=len(ev), trajectory=traj)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{task}__{split}__s{seed}__lr{lr:g}.json", "w"), indent=2)
    print(f"  RESULT {task}/{split}/s{seed}/lr{lr:g}: ev_auroc={res['ev_auroc']:.4f} "
          f"ev_mcc={res['ev_mcc']:.4f} @ep{best['ep']} params={n_par/1e6:.2f}M", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["Host_Tropism"])
    ap.add_argument("--splits", nargs="+", default=["cluster_disjoint", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split_dir", default=None)
    a = ap.parse_args()
    if a.split_dir:
        globals()["SPL"] = a.split_dir
    for t in a.tasks:
        for s in a.splits:
            for lr in a.lrs:
                for sd in a.seeds:
                    run(t, s, sd, lr, a.epochs, a.device)
