"""
Model-agnostic FULL fine-tuning on the EXACT v2 composition-confound splits.

Purpose: test whether the single positive in the programme (Evo LoRA on
Host_Tropism cluster-disjoint, +0.0139 over kmer3-6) replicates in other gLM
architectures, under a protocol that fixes the three flaws in the Evo runs:

  1. ev (the reported, cluster-disjoint partition) is evaluated at EVERY epoch, so
     the trajectory is visible instead of a single point at the best-iv step.
  2. LR grid is swept and reported, not pinned at its ceiling.
  3. Real early stopping (min_delta 5e-4, patience 6) with headroom, so runs are
     not truncated by the epoch cap.

Also reports `ev_auroc_oracle` = max ev over epochs. The gap between the honest
(best-iv-selected) ev and the oracle ev quantifies how much of any apparent win is
checkpoint-selection luck.

Baselines on these exact splits (from kmer_results_v2.csv):
  Host_Tropism     random 0.9213   cluster_disjoint 0.8034
  Pathogenecity    random 0.9685   cluster_disjoint 0.8044
  Transmissibility random 0.9238   cluster_disjoint 0.7395
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")

SPL = P.SPLITS_V2
EXT_KMER = {}
OUT = P.sub("hvue_glm")
KMER = {("Host_Tropism","random"):0.9213, ("Host_Tropism","cluster_disjoint"):0.8034,
        ("Pathogenecity","random"):0.9685, ("Pathogenecity","cluster_disjoint"):0.8044,
        ("Transmissibility","random"):0.9238, ("Transmissibility","cluster_disjoint"):0.7395}
MODELS = {
  "hyenadna":  ("LongSafari/hyenadna-medium-160k-seqlen-hf", "AutoModel", 256),
  "gena_lm":   ("AIRI-Institute/gena-lm-bert-base-t2t",      "AutoModel", 768),
  "nt_v2_500m":("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species","AutoModelForMaskedLM",1024),
}


class Wrap(nn.Module):
    def __init__(self, mid, how, hid, random_init=False, regime="full", lora_rank=16):
        super().__init__()
        cls = {"AutoModel": AutoModel, "AutoModelForMaskedLM": AutoModelForMaskedLM}[how]
        if random_init:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
            self.body = cls.from_config(cfg, trust_remote_code=True)
        else:
            self.body = cls.from_pretrained(mid, trust_remote_code=True)
        self.how = how; self.regime = regime
        if regime == "probe":
            for prm in self.body.parameters():
                prm.requires_grad = False
        elif regime == "lora":
            from peft import LoraConfig, get_peft_model
            tgts = [n for n, m_ in self.body.named_modules() if isinstance(m_, nn.Linear)]
            if not tgts:
                raise RuntimeError("no nn.Linear found for LoRA")
            self.body = get_peft_model(self.body, LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
                bias="none", target_modules=tgts))
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, ids, mask):
        kw = dict(input_ids=ids)
        if mask is not None:
            kw["attention_mask"] = mask
        o = self.body(**kw, output_hidden_states=True)
        h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o.last_hidden_state
        if mask is None:
            pooled = h.mean(1)
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float()).squeeze(-1)


def tokenize(tok, seqs, maxlen):
    e = tok(list(seqs), return_tensors="pt", padding="max_length",
            truncation=True, max_length=maxlen)
    return e["input_ids"], e.get("attention_mask")


def run(mkey, task, split, seed, lr, epochs, dev, maxlen, bs, rand=False, regime="full", lora_rank=16):
    mid, how, hid = MODELS[mkey]
    torch.manual_seed(seed); np.random.seed(seed)
    df = pd.read_parquet(f"{SPL}/{task}__{split}.parquet")
    tr_all = df[df.partition == "train"].reset_index(drop=True)
    ev = df[df.partition == "val"].reset_index(drop=True)
    rng = np.random.default_rng(seed); ivm = np.zeros(len(tr_all), bool)
    for c in tr_all.label.unique():
        idx = np.where(tr_all.label.values == c)[0]; rng.shuffle(idx)
        ivm[idx[:int(0.15 * len(idx))]] = True
    tr, ivd = tr_all[~ivm].reset_index(drop=True), tr_all[ivm].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    Itr, Mtr = tokenize(tok, tr.sequence, maxlen); ytr = torch.tensor(tr.label.values, dtype=torch.float32)
    Iiv, Miv = tokenize(tok, ivd.sequence, maxlen); yiv = ivd.label.values.astype(int)
    Iev, Mev = tokenize(tok, ev.sequence, maxlen);  yev = ev.label.values.astype(int)

    m = Wrap(mid, how, hid, random_init=rand, regime=regime, lora_rank=lora_rank).to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    train_p = [p for p in m.parameters() if p.requires_grad]
    n_tr_par = sum(p.numel() for p in train_p)
    print(f"  [{mkey}/{regime}] params={n_par/1e6:.2f}M trainable={n_tr_par/1e6:.3f}M "
          f"({100*n_tr_par/n_par:.2f}%)", flush=True)
    opt = torch.optim.AdamW(train_p, lr=lr, weight_decay=0.01)
    total = max(1, epochs * (len(Itr) // bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    def score(I, M):
        m.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(I), bs * 2):
                mm = M[i:i + bs * 2].to(dev) if M is not None else None
                out.append(m(I[i:i + bs * 2].to(dev), mm).float().cpu().numpy())
        m.train(); return np.concatenate(out)

    best = dict(iv=0.0, ep=0, sev=None, siv=None); traj = []; noimp = 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Itr)); t0 = time.time()
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            mm = Mtr[b].to(dev) if Mtr is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(m(Itr[b].to(dev), mm), ytr[b].to(dev))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(train_p, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        siv, sev = score(Iiv, Miv), score(Iev, Mev)
        a_iv, a_ev = roc_auc_score(yiv, siv), roc_auc_score(yev, sev)
        traj.append(dict(epoch=ep, iv_auroc=round(a_iv, 4), ev_auroc=round(a_ev, 4)))
        print(f"  [{mkey}/{task}/{split}/s{seed}/lr{lr:g}] ep{ep:3d} iv={a_iv:.4f} ev={a_ev:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if a_iv > best["iv"] + 5e-4:
            best.update(iv=a_iv, ep=ep, sev=sev.copy(), siv=siv.copy()); noimp = 0
        else:
            noimp += 1
            if noimp >= 6:
                print(f"  early stop @ ep{ep}", flush=True); break
    fpr, tpr, th = roc_curve(yiv, best["siv"]); thr = float(th[np.argmax(tpr - fpr)])
    ev_a = float(roc_auc_score(yev, best["sev"]))
    km = KMER[(task, split)]
    oracle = float(max(t["ev_auroc"] for t in traj))
    res = dict(model=(mkey+"_randinit" if rand else mkey), pretrained=(not rand),
               regime=regime, n_trainable=n_tr_par, trainable_frac=round(n_tr_par/n_par, 5), hf_id=mid, n_params=n_par, task=task, split=split, seed=seed, lr=lr,
               best_epoch=best["ep"], epochs_run=len(traj), iv_auroc=round(float(best["iv"]), 4),
               ev_auroc=round(ev_a, 4),
               ev_mcc=round(float(matthews_corrcoef(yev, (best["sev"] >= thr).astype(int))), 4),
               kmer_auroc=km, excess=round(ev_a - km, 4),
               ev_auroc_oracle=round(oracle, 4), excess_oracle=round(oracle - km, 4),
               ev_traj_min=round(min(t["ev_auroc"] for t in traj), 4),
               ev_traj_range=round(oracle - min(t["ev_auroc"] for t in traj), 4),
               n_train=len(tr), n_iv=len(ivd), n_ev=len(ev), trajectory=traj)
    os.makedirs(OUT, exist_ok=True)
    tagm = (mkey+"_randinit" if rand else mkey) + "__" + regime
    json.dump(res, open(f"{OUT}/{tagm}__{task}__{split}__s{seed}__lr{lr:g}.json", "w"), indent=2)
    print(f"  RESULT {tagm}/{task}/{split}/s{seed}/lr{lr:g}: ev={ev_a:.4f} kmer={km:.4f} "
          f"excess={ev_a-km:+.4f} | oracle_ev={oracle:.4f} ({oracle-km:+.4f}) "
          f"| ev range over epochs={res['ev_traj_range']:.4f}", flush=True)
    del m; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tasks", nargs="+", default=["Host_Tropism"])
    ap.add_argument("--splits", nargs="+", default=["cluster_disjoint", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-5, 5e-5])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--maxlen", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--regime", default="full", choices=["probe","lora","full"])
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--kmer_json", default=None)
    a = ap.parse_args()
    ml = a.maxlen or {"hyenadna": 1024, "gena_lm": 256, "nt_v2_500m": 176}[a.model]
    if a.split_dir:
        globals()["SPL"] = a.split_dir
    if a.kmer_json:
        for k, v in json.load(open(a.kmer_json)).items():
            t, sp = k.split("__", 1); KMER[(t, sp)] = v[0]
    for t in a.tasks:
        for s in a.splits:
            for lr in a.lrs:
                for sd in a.seeds:
                    try:
                        run(a.model, t, s, sd, lr, a.epochs, a.device, ml, a.bs, a.random_init, a.regime, a.lora_rank)
                    except Exception as e:
                        print(f"  FAILED {a.model}/{t}/{s}/s{sd}/lr{lr:g}: {type(e).__name__}: {e}", flush=True)
