
"""
PHASE 2 (mandatory): validate OUR OWN frozen-probe pipeline on a task where a genuine FM
advantage is known to exist.

Every splice-site claim so far compares OUR baseline against PUBLISHED FM numbers. That does
not establish that OUR pipeline can detect an FM advantage. This runs the SAME frozen-probe
code path used for the viral experiments (mean-pooled frozen embeddings -> standardised LR
head, C selected on a held-out dev slice of TRAIN, test scored once) on the NT
chromosome-disjoint splice tasks, against the SAME k-mer and CNN baselines.

Expected if the pipeline is sound: large positive FM - baseline gap on splice.
Observed on viral tasks with identical code: no advantage.
That contrast is the control the negative result rests on.

Data: /data/nvidia/data/ntv3/{splice_sites_all,_acceptors,_donors}/{train,test}.parquet
      columns [sequence, name, label, task]; 600 bp; 3 classes for _all, 2 for donors/acceptors.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import paths as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
try:
    import transformers; transformers.logging.set_verbosity_error()
except Exception: pass

NT_DIR = os.environ.get("VB_NT_DIR", "/data/nvidia/data/ntv3")
OUT = P.sub("splice_positive_control")
MODELS = {
 "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "AutoModelForMaskedLM", 1024, 6),
 "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t", "AutoModel", 768, 6),
 "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", "AutoModel", 256, 1),
 "lucavirus":  ("LucaGroup/LucaVirus-default-step3.8M", "AutoModel", 2560, 1),
}
CODE = np.full(256, 255, np.uint8); MAPI = np.full(256, 4, np.int64)
for i, c in enumerate("ACGT"):
    CODE[ord(c)] = i; CODE[ord(c.lower())] = i; MAPI[ord(c)] = i; MAPI[ord(c.lower())] = i

def kmer_feats(seqs, ks):
    out = []
    for k in ks:
        V = 4**k; X = np.zeros((len(seqs), V), np.float32)
        pw = (4**np.arange(k-1, -1, -1)).astype(np.int64)
        for r, s in enumerate(seqs):
            c = CODE[np.frombuffer(s.encode(), np.uint8)]; ok = c != 255
            if ok.sum() < k: continue
            c = c.astype(np.int64)
            if len(c)-k+1 <= 0: continue
            w = np.lib.stride_tricks.sliding_window_view(c, k)
            v = np.lib.stride_tricks.sliding_window_view(ok, k).all(1)
            idx = (w @ pw)[v]
            if idx.size: np.add.at(X[r], idx, 1.0); X[r] /= idx.size
        out.append(X)
    return np.hstack(out)

class CNN(nn.Module):
    def __init__(self, ncls, ch=128, nb=4, drop=0.2):
        super().__init__(); self.emb = nn.Embedding(5, 16); L, cin = [], 16
        for i in range(nb):
            L += [nn.Conv1d(cin, ch, 9, padding=2**i*4, dilation=2**i), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop)]
            cin = ch
        self.conv = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Linear(2*ch, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, ncls))
    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

def enc(seqs, L):
    X = np.full((len(seqs), L), 4, np.int64)
    for i, s in enumerate(seqs):
        a = MAPI[np.frombuffer(s[:L].encode(), np.uint8)]; X[i, :len(a)] = a
    return X

def met(y, p):
    return dict(mcc=round(float(matthews_corrcoef(y, p)), 4),
                macro_f1=round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
                accuracy=round(float(accuracy_score(y, p)), 4))

@torch.no_grad()
def embed(model, tok, seqs, ml, bs, dev):
    out = []
    for i in range(0, len(seqs), bs):
        e = tok(list(seqs[i:i+bs]), return_tensors="pt", padding="max_length", truncation=True, max_length=ml)
        ids = e["input_ids"].to(dev); am = e.get("attention_mask")
        kw = dict(input_ids=ids)
        if am is not None: kw["attention_mask"] = am.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**kw, output_hidden_states=True)
        h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o.last_hidden_state
        if am is None: pooled = h.mean(1)
        else:
            mm = am.to(dev).unsqueeze(-1).to(h.dtype); pooled = (h*mm).sum(1)/mm.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.vstack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["splice_sites_all","splice_sites_acceptors","splice_sites_donors"])
    ap.add_argument("--models", nargs="+", default=["nt_v2_500m"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    tr = pd.read_parquet(f"{NT_DIR}/{a.task}/train.parquet")
    te = pd.read_parquet(f"{NT_DIR}/{a.task}/test.parquet")
    ncls = int(pd.concat([tr, te]).label.nunique())
    rng = np.random.default_rng(0); m = np.zeros(len(tr), bool)
    m[rng.permutation(len(tr))[:int(0.15*len(tr))]] = True
    dv, tr2 = tr[m].reset_index(drop=True), tr[~m].reset_index(drop=True)
    L = len(tr.sequence.iloc[0])
    print(f"{a.task}: classes={ncls} train={len(tr2)} dev={len(dv)} test={len(te)} seqlen={L}", flush=True)
    res = dict(task=a.task, n_classes=ncls, n_train=len(tr2), n_dev=len(dv), n_test=len(te), seqlen=L)

    # ---- k-mer baseline: SAME LR protocol as the FM probe, C on dev only ----
    for ks, nm in [((3,4,5), "kmer3-5"), ((3,4,5,6), "kmer3-6")]:
        t = time.time()
        Xtr, Xdv, Xte = (kmer_feats(d.sequence.tolist(), ks) for d in (tr2, dv, te))
        sc = StandardScaler().fit(Xtr)
        Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
        best = (-1, None, None)
        for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=1500, n_jobs=-1).fit(Xtr, tr2.label)
            s = matthews_corrcoef(dv.label, clf.predict(Xdv))
            if s > best[0]: best = (s, C, clf)
        res[nm] = dict(**met(te.label, best[2].predict(Xte)), C=best[1], dev_mcc=round(best[0], 4))
        print(f"  {nm:<9} C={best[1]:<7} test MCC={res[nm]['mcc']:.4f} ({time.time()-t:.0f}s)", flush=True)

    # ---- CNN: supervised, separate baseline family ----
    Itr, Idv, Ite = (torch.from_numpy(enc(d.sequence.tolist(), L)) for d in (tr2, dv, te))
    ytr = torch.tensor(tr2.label.values, dtype=torch.long); runs = []
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        mdl = CNN(ncls).to(a.device); opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        def pr(I):
            mdl.eval(); o = []
            with torch.no_grad():
                for i in range(0, len(I), 256): o.append(mdl(I[i:i+256].to(a.device)).argmax(-1).cpu().numpy())
            mdl.train(); return np.concatenate(o)
        bst = (-1, 0, None); noimp = 0
        for ep in range(1, a.epochs+1):
            perm = torch.randperm(len(Itr))
            for i in range(0, len(perm)-a.bs+1, a.bs):
                b = perm[i:i+a.bs]
                loss = F.cross_entropy(mdl(Itr[b].to(a.device)), ytr[b].to(a.device))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
            sch.step()
            s = matthews_corrcoef(dv.label, pr(Idv))
            if s > bst[0]+5e-4: bst = (s, ep, pr(Ite).copy()); noimp = 0
            else:
                noimp += 1
                if noimp >= 6: break
        runs.append(dict(**met(te.label, bst[2]), seed=seed, best_epoch=bst[1], dev_mcc=round(bst[0], 4)))
        print(f"  CNN s{seed}: test MCC={runs[-1]['mcc']:.4f} @ep{bst[1]}", flush=True)
    res["cnn"] = dict(runs=runs, mean_mcc=round(float(np.mean([r['mcc'] for r in runs])), 4),
                      sd_mcc=round(float(np.std([r['mcc'] for r in runs], ddof=1)), 4))

    # ---- FM frozen probe: IDENTICAL protocol to the viral experiments ----
    for mk in a.models:
        mid, how, hid, bpt = MODELS[mk]
        ml = L // bpt + 2
        cls_ = {"AutoModel": AutoModel, "AutoModelForMaskedLM": AutoModelForMaskedLM}[how]
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        mdl = cls_.from_pretrained(mid, trust_remote_code=True).to(a.device).eval()
        t = time.time()
        Etr, Edv, Ete = (embed(mdl, tok, d.sequence.values, ml, a.bs, a.device) for d in (tr2, dv, te))
        del mdl; torch.cuda.empty_cache()
        sc = StandardScaler().fit(Etr)
        Etr, Edv, Ete = sc.transform(Etr), sc.transform(Edv), sc.transform(Ete)
        best = (-1, None, None)
        for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=1500, n_jobs=-1).fit(Etr, tr2.label)
            s = matthews_corrcoef(dv.label, clf.predict(Edv))
            if s > best[0]: best = (s, C, clf)
        res[mk] = dict(**met(te.label, best[2].predict(Ete)), C=best[1], dev_mcc=round(best[0], 4),
                       regime="frozen_probe", effective_context_bp=ml*bpt, runtime_s=round(time.time()-t))
        bb = max(res["kmer3-5"]["mcc"], res["kmer3-6"]["mcc"], res["cnn"]["mean_mcc"])
        print(f"  {mk:<12} frozen MCC={res[mk]['mcc']:.4f} | best baseline={bb:.4f} | "
              f"FM-baseline={res[mk]['mcc']-bb:+.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{a.task}__{'_'.join(a.models)}.json", "w"), indent=2)
    print(f"  wrote {OUT}/{a.task}__{'_'.join(a.models)}.json", flush=True)

if __name__ == "__main__":
    main()
