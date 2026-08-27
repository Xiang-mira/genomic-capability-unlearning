
"""
BASELINE CAPACITY CEILING sweep.

Every "a small CNN matches the FM" claim in this project rests on ONE architecture at
0.64M params. That is an arbitrary point, not a ceiling. This sweeps architecture family
x capacity (0.1M -> ~10M) so we can state whether the supervised baseline is saturated.

Two possible outcomes, both scientifically useful:
  - CNN plateaus well below 10M  -> baseline is saturated, FM comparisons stand as fair
  - CNN keeps improving to 10M   -> our published baseline was UNDER-POWERED, and any FM
                                    win inside the improvement margin is void

Architectures:
  dilated : the incumbent (exponentially dilated conv stack, max+mean pool)
  unet    : encoder/decoder with skip connections, strided down / transposed up
  resnet  : residual conv blocks, no dilation

Discipline: architecture AND capacity are selected on DEV only; test is scored once per
(arch, capacity) cell and the selection is reported separately so the reader can see both
the dev-selected number and the full curve.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import paths as P
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

OUT = P.sub("capacity_sweep")
NT_DIR = os.environ.get("VB_NT_DIR", "/data/nvidia/data/ntv3")
MAPI = np.full(256, 4, np.int64)
for i, c in enumerate("ACGT"):
    MAPI[ord(c)] = i; MAPI[ord(c.lower())] = i

# ---------------- architectures ----------------
class Dilated(nn.Module):
    def __init__(self, ncls, ch, nb, drop=0.2):
        super().__init__(); self.emb = nn.Embedding(5, 16); L, cin = [], 16
        for i in range(nb):
            L += [nn.Conv1d(cin, ch, 9, padding=2**min(i,8)*4, dilation=2**min(i,8)),
                  nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop)]; cin = ch
        self.conv = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Linear(2*ch, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

class UNet(nn.Module):
    """Encoder/decoder with skip connections -- multi-scale receptive field without
    exponential dilation, the natural 'more capacity' alternative for long sequences."""
    def __init__(self, ncls, ch, depth, drop=0.2):
        super().__init__(); self.emb = nn.Embedding(5, 16)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        cin = 16
        chans = [ch * (2**min(i, 3)) for i in range(depth)]
        for c in chans:
            self.down.append(nn.Sequential(nn.Conv1d(cin, c, 9, padding=4), nn.BatchNorm1d(c),
                                           nn.GELU(), nn.Conv1d(c, c, 9, stride=4, padding=4),
                                           nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            cin = c
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.Sequential(nn.Conv1d(chans[i] + chans[i-1], chans[i-1], 9, padding=4),
                                         nn.BatchNorm1d(chans[i-1]), nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Sequential(nn.Linear(2*chans[0], 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.emb(x).transpose(1, 2); skips = []
        for d in self.down:
            skips.append(h); h = d(h)
        for k, u in enumerate(self.up):
            s = skips[-(k+1)]
            h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = u(torch.cat([h, s], 1))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

class ResBlk(nn.Module):
    def __init__(self, c, drop):
        super().__init__()
        self.b = nn.Sequential(nn.Conv1d(c, c, 9, padding=4), nn.BatchNorm1d(c), nn.GELU(),
                               nn.Dropout(drop), nn.Conv1d(c, c, 9, padding=4), nn.BatchNorm1d(c))
    def forward(self, x): return F.gelu(x + self.b(x))

class ResNet(nn.Module):
    def __init__(self, ncls, ch, nb, drop=0.2):
        super().__init__(); self.emb = nn.Embedding(5, 16)
        self.stem = nn.Sequential(nn.Conv1d(16, ch, 9, padding=4), nn.BatchNorm1d(ch), nn.GELU())
        self.body = nn.Sequential(*[ResBlk(ch, drop) for _ in range(nb)])
        self.head = nn.Sequential(nn.Linear(2*ch, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, ncls))
    def forward(self, x):
        h = self.body(self.stem(self.emb(x).transpose(1, 2)))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

def build(arch, ncls, ch, nb):
    return {"dilated": Dilated, "unet": UNet, "resnet": ResNet}[arch](ncls, ch, nb)

# capacity ladder: (arch, ch, blocks) chosen to span ~0.1M -> ~10M
LADDER = [
    # (arch, ch, blocks) calibrated to span ~0.04M -> ~10M params.
    # dilated ch=128 nb=5 (0.68M) is the INCUMBENT baseline used in every prior result.
    ("dilated", 32, 3), ("dilated", 64, 4), ("dilated", 128, 5), ("dilated", 256, 5), ("dilated", 448, 6),
    ("unet",    24, 3), ("unet",   48, 3), ("unet",   112, 3), ("unet",    64, 4),
    ("resnet",  64, 3), ("resnet", 128, 4), ("resnet", 256, 5), ("resnet", 320, 5),
]
INCUMBENT = ("dilated", 128, 5)

def enc(seqs, L):
    X = np.full((len(seqs), L), 4, np.int64)
    for i, s in enumerate(seqs):
        a = MAPI[np.frombuffer(s[:L].encode(), np.uint8)]; X[i, :len(a)] = a
    return X

def load(ds, task, level, seqcap, min_count=1):
    if ds == "splice":
        tr = pd.read_parquet(f"{NT_DIR}/{task}/train.parquet")
        te = pd.read_parquet(f"{NT_DIR}/{task}/test.parquet")
        rng = np.random.default_rng(0); m = np.zeros(len(tr), bool)
        m[rng.permutation(len(tr))[:int(.15*len(tr))]] = True
        return tr[~m].reset_index(drop=True), tr[m].reset_index(drop=True), te, "label", "mcc"
    if ds == "hvue":
        # allow an explicit split file so the ladder can run on the homology-strict splits,
        # not just splits_ungated/*identity_disjoint_hsd0
        _sd = os.environ.get("VB_SPLIT_DIR") or P.sub("splits_ungated")
        _sx = os.environ.get("VB_SPLIT_SUFFIX", "identity_disjoint_hsd0")
        d = pd.read_parquet(f"{_sd}/{task}__{_sx}.parquet")
        trall = d[d.partition == "train"].reset_index(drop=True)
        te = d[d.partition == "val"].reset_index(drop=True)
        # group-disjoint dev, so capacity/arch selection faces the same homology holdout as test
        from sklearn.model_selection import GroupShuffleSplit
        gi, di = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                      .split(trall, groups=trall.group.values))
        return (trall.iloc[gi].reset_index(drop=True), trall.iloc[di].reset_index(drop=True),
                te, "label", "auroc")
    if ds == "geneb":
        # GENEB ships tasks/<id>/{train,test}.csv with columns text,label and NO dev split.
        # Dev carve is a stratified 15% of train at seed 42 -- identical to
        # scripts/geneb/fair_kmer_sentinel.py and scripts/track_a_benchmarks/geneb_finetune.py,
        # so the CNN ladder, the fair k-mer and the FT arm all select on the same rows.
        G = os.environ.get("VB_GENEB_DIR", "/data/nvidia/geneb_data/tasks")
        tr_all = pd.read_csv(f"{G}/{task}/train.csv").rename(columns={"text": "sequence"})
        te = pd.read_csv(f"{G}/{task}/test.csv").rename(columns={"text": "sequence"})
        from sklearn.model_selection import train_test_split
        i_tr, i_dv = train_test_split(np.arange(len(tr_all)), test_size=0.15,
                                      stratify=tr_all.label.values, random_state=42)
        return (tr_all.iloc[i_tr].reset_index(drop=True),
                tr_all.iloc[i_dv].reset_index(drop=True), te, "label", "mcc")
    if ds == "virobench":
        # identical construction to virobench_baselines.py so the ladder is directly comparable
        import virobench_baselines as VBB
        tr, dv, te = (VBB.load("ALL", "times", x) for x in ("train", "val", "test"))
        # min_count MUST match virobench_frozen_probe.py (default 1) or the label space
        # differs and the ladder is not comparable to the frozen-probe numbers
        keep = tr[level].value_counts()
        keep = set(keep[keep >= min_count].index) & set(te[level].dropna())
        tr, dv, te = (d[d[level].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
        ixm = {c: i for i, c in enumerate(sorted(keep))}
        for d in (tr, dv, te):
            d["label"] = d[level].map(ixm)
        return tr, dv, te, "label", "macro_f1"
    raise ValueError(ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["splice","hvue","virobench","geneb"])
    ap.add_argument("--task", default="ALL_times")
    ap.add_argument("--level", default="family")
    ap.add_argument("--seqlen", type=int, default=0, help="0 = native length")
    ap.add_argument("--min_count", type=int, default=1, help="virobench: match frozen probe (1)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,43])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-4, 3e-3],
                    help="LR grid; the FMs get one, so the baseline must too")
    ap.add_argument("--topk", type=int, default=3, help="architectures carried into the LR sweep"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    tr, dv, te, ycol, metric = load(a.dataset, a.task, a.level, a.seqlen, a.min_count)
    L = a.seqlen or int(pd.concat([tr, dv, te]).sequence.str.len().max())
    ncls = int(pd.concat([tr,dv,te])[ycol].nunique())
    print(f"{a.dataset}/{a.task}: classes={ncls} train={len(tr)} dev={len(dv)} test={len(te)} L={L} metric={metric}", flush=True)
    Itr = torch.from_numpy(enc(tr.sequence.tolist(), L)); ytr = torch.tensor(tr[ycol].values, dtype=torch.long)
    Idv = torch.from_numpy(enc(dv.sequence.tolist(), L)); ydv = dv[ycol].values.astype(int)
    Ite = torch.from_numpy(enc(te.sequence.tolist(), L)); yte = te[ycol].values.astype(int)

    def score(y, p, prob=None):
        if metric == "auroc": return float(roc_auc_score(y, prob))
        if metric == "macro_f1": return float(f1_score(y, p, average="macro", zero_division=0))
        return float(matthews_corrcoef(y, p))
    def run_cell(arch, ch, nb, lr, seeds):
        """Train one (architecture, lr) cell over `seeds`; return per-seed dev/test."""
        per = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m = build(arch, ncls, ch, nb).to(a.device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
            def pred(I):
                m.eval(); pr=[]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(0, len(I), 256):
                        pr.append(torch.softmax(m(I[i:i+256].to(a.device)).float(), -1).cpu().numpy())
                m.train(); pr=np.vstack(pr); return pr.argmax(1), pr
            best=(-1,0,None,None); noimp=0
            try:
                for ep in range(1, a.epochs+1):
                    perm = torch.randperm(len(Itr))
                    for i in range(0, len(perm)-a.bs+1, a.bs):
                        b = perm[i:i+a.bs]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = F.cross_entropy(m(Itr[b].to(a.device)), ytr[b].to(a.device))
                        opt.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                    sch.step()
                    pd_, pp_ = pred(Idv); sc = score(ydv, pd_, pp_[:,1] if ncls==2 else None)
                    if sc > best[0]+5e-4:
                        pt_, ptp_ = pred(Ite); best=(sc, ep, pt_, ptp_); noimp=0
                    else:
                        noimp += 1
                        if noimp >= 6: break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); del m; return None
            ts = score(yte, best[2], best[3][:,1] if ncls==2 else None)
            per.append(dict(seed=seed, dev=round(best[0],4), test=round(ts,4), best_epoch=best[1]))
            del m; torch.cuda.empty_cache()
        return per

    # ---- STAGE 1: architecture search at the base LR, 1 seed ----
    stage1 = []
    for arch, ch, nb in LADDER:
        npar = sum(p.numel() for p in build(arch, ncls, ch, nb).parameters())
        per = run_cell(arch, ch, nb, a.lrs[0], a.seeds[:1])
        if per is None:
            print(f"  [s1] {arch}-{ch}x{nb} ({npar/1e6:.2f}M): OOM", flush=True); continue
        stage1.append(dict(arch=arch, ch=ch, blocks=nb, params=npar, params_M=round(npar/1e6,3),
                           lr=a.lrs[0], dev=per[0]['dev'], test=per[0]['test'], runs=per))
        print(f"  [s1] {arch:<8} ch={ch:<4} nb={nb} {npar/1e6:6.2f}M lr={a.lrs[0]:<7g} "
              f"dev={per[0]['dev']:.4f} test={per[0]['test']:.4f}", flush=True)
    topk = sorted(stage1, key=lambda r: -r['dev'])[:a.topk]
    print(f"  [s1] top-{a.topk} by dev: {[(r['arch'], r['params_M']) for r in topk]}", flush=True)

    # ---- STAGE 2: LR grid on the top-K architectures, all seeds ----
    # The FMs get an LR sweep; without this the CNN baseline is under-tuned by construction
    # and any FM margin is confounded with baseline tuning effort.
    rows = list(stage1)
    for r0 in topk:
        for lr in a.lrs:
            seeds = a.seeds if lr != a.lrs[0] else a.seeds[1:]   # stage 1 already did (lr0, seed0)
            if not seeds: continue
            per = run_cell(r0['arch'], r0['ch'], r0['blocks'], lr, seeds)
            if per is None: continue
            if lr == a.lrs[0]: per = r0['runs'] + per
            dv_m = float(np.mean([p['dev'] for p in per])); te_m = float(np.mean([p['test'] for p in per]))
            rows.append(dict(arch=r0['arch'], ch=r0['ch'], blocks=r0['blocks'], params=r0['params'],
                             params_M=r0['params_M'], lr=lr, dev=round(dv_m,4), test=round(te_m,4),
                             test_sd=round(float(np.std([p['test'] for p in per], ddof=1)),4) if len(per)>1 else None,
                             runs=per))
            print(f"  [s2] {r0['arch']:<8} {r0['params_M']:>7}M lr={lr:<7g} "
                  f"dev={dv_m:.4f} test={te_m:.4f}", flush=True)
    best_by_dev = max(rows, key=lambda r: r['dev'])
    res = dict(dataset=a.dataset, task=a.task, metric=metric, n_classes=ncls, seqlen=L,
               n_train=len(tr), n_dev=len(dv), n_test=len(te), seeds=a.seeds,
               ladder=rows,
               dev_selected=dict(arch=best_by_dev['arch'], params_M=best_by_dev['params_M'], lr=best_by_dev.get('lr'),
                                 dev=best_by_dev['dev'], test=best_by_dev['test']),
               oracle_best_test=max(r['test'] for r in rows),
               incumbent=next(({k: r[k] for k in ('arch','params_M','dev','test')} for r in rows
                               if (r['arch'], r['ch'], r['blocks']) == INCUMBENT), None))
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.dataset}__{a.task}" + (f"__{os.environ['VB_SPLIT_SUFFIX']}" if os.environ.get('VB_SPLIT_SUFFIX') else "") + (f"__{a.level}" if a.dataset == "virobench" else "")
    json.dump(res, open(f"{OUT}/{tag}.json","w"), indent=2)
    print(f"\n  DEV-SELECTED: {best_by_dev['arch']} @ {best_by_dev['params_M']}M lr={best_by_dev.get('lr')} -> test {best_by_dev['test']:.4f}")
    print(f"  oracle best test across ladder: {res['oracle_best_test']:.4f} (report as oracle, not as the baseline)")
    print(f"  wrote {OUT}/{tag}.json", flush=True)

if __name__ == "__main__":
    main()
