"""
gLMs on ViroBench taxonomy classification -- the one benchmark family in this repo
where a real capability target was never tested (HANDOFF.md P3): only k-mer/CNN
baselines exist in virobench_baselines.py, no gLM has been run.

Use --split times (the genuinely temporal-disjoint split; --split genus is NOT
genus-disjoint, 82-84% genus overlap, see virobench_baselines.py's own docstring
and this session's independent leakage audit -- do not use genus for a capability
claim). Sequences are full viral genomes (median 43kb); every model is truncated
to a fixed token budget and the resulting effective context (in bp) is recorded
explicitly in the result JSON, matching virobench_baselines.py's own "context"
field for the k-mer/CNN baselines so the two are directly comparable.

Same discipline as gue_glm.py: regime in {probe,lora,full}, model selected on the
ViroBench dev(val) split, macro-F1/MCC reported on test, test evaluated every
epoch so the trajectory is visible. One result JSON per (model,regime,seed,lr) --
resuming via skip-if-exists is safe here (unlike an aggregated-multi-seed file).
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
from virobench_baselines import load
warnings.filterwarnings("ignore")
D = P.VIRO_DIR
OUT = P.sub("virobench_glm")

# max_length in TOKENS, chosen to match virobench_baselines.py's own effective-context
# convention (README: "NT-v2 ~12kb, GENA-LM ~2-4kb, HyenaDNA 160kb" of context) but
# capped at 20000bp-equivalent for hyenadna to keep full-genome runs tractable on a
# single GPU -- same cap virobench_baselines.py's CNN uses (--cnn_len 20000), so the
# gLM and the CNN baseline see the same amount of sequence.
MODELS = {
    "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", "AutoModel", 256, 20000),
    "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t",      "AutoModel", 768, 512),
    # nt_v2_500m's custom modeling code (modeling_esm.py) computes attention eagerly
    # (no fused/flash kernel) -- O(n^2) memory. 2048 tokens x bs=8 OOM'd a 96GB GPU
    # even with a 500M model. 1024 tokens is the practical ceiling here; gradient
    # checkpointing + a smaller batch are used below to make even that fit.
    "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "AutoModelForMaskedLM", 1024, 1024),
}


class Wrap(nn.Module):
    def __init__(self, mid, how, hid, ncls, regime="full", lora_rank=16):
        super().__init__()
        cls = {"AutoModel": AutoModel, "AutoModelForMaskedLM": AutoModelForMaskedLM}[how]
        self.body = cls.from_pretrained(mid, trust_remote_code=True)
        if regime != "probe":
            try:
                self.body.gradient_checkpointing_enable()
            except ValueError:
                pass  # model doesn't declare support even though it can OOM without it
        if regime == "probe":
            for p in self.body.parameters(): p.requires_grad = False
        elif regime == "lora":
            from peft import LoraConfig, get_peft_model
            tg = [n for n, mm in self.body.named_modules() if isinstance(mm, nn.Linear)]
            self.body = get_peft_model(self.body, LoraConfig(r=lora_rank, lora_alpha=2*lora_rank,
                                                             lora_dropout=0.0, bias="none", target_modules=tg))
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, ncls))

    def forward(self, ids, mask):
        kw = dict(input_ids=ids)
        if mask is not None: kw["attention_mask"] = mask
        o = self.body(**kw, output_hidden_states=True)
        h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o.last_hidden_state
        if mask is None:
            pooled = h.mean(1)
        else:
            m = mask.unsqueeze(-1).to(h.dtype); pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float())


def metrics(y, p):
    return dict(macro_f1=round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
                accuracy=round(float(accuracy_score(y, p)), 4),
                mcc=round(float(matthews_corrcoef(y, p)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--mod", default="DNA"); ap.add_argument("--split", default="times")
    ap.add_argument("--level", default="family"); ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--regime", default="full", choices=["probe", "lora", "full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-5])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=None); ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    mid, how, hid, default_ml = MODELS[a.model]
    ml = a.maxlen or default_ml

    tr, dv, te = (load(a.mod, a.split, s) for s in ["train", "val", "test"])
    lv = a.level
    keep = tr[lv].value_counts(); keep = set(keep[keep >= a.min_count].index) & set(te[lv].dropna())
    tr, dv, te = (d[d[lv].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
    cls = sorted(keep); ix = {c: i for i, c in enumerate(cls)}
    for d in (tr, dv, te): d["y"] = d[lv].map(ix)
    if len(tr) > a.cap: tr = tr.sample(a.cap, random_state=0).reset_index(drop=True)
    ncls = len(cls)

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    def tk(seqs):
        e = tok(list(seqs), return_tensors="pt", padding="max_length", truncation=True, max_length=ml)
        return e["input_ids"], e.get("attention_mask")

    Itr, Mtr = tk(tr.sequence); Idv, Mdv = tk(dv.sequence); Ite, Mte = tk(te.sequence)
    ytr = torch.tensor(tr.y.values); ydv = dv.y.values; yte = te.y.values
    # approx effective context in bp: HyenaDNA is byte-level (~1 char/token), NT-v2 is
    # 6-mer (~6 char/token), GENA-LM is BPE (approx, varies) -- report both token count
    # and the approx bp figure so this isn't hidden the way README rule 4 requires.
    bp_per_tok = {"hyenadna": 1, "gena_lm": 6, "nt_v2_500m": 6}[a.model]
    eff_ctx_bp = ml * bp_per_tok
    print(f"{a.model}/{a.mod}/{a.split}/{lv}: n_tr={len(tr)} classes={ncls} maxlen_tok={ml} "
          f"eff_context~{eff_ctx_bp}bp median_genome={int(tr.sequence.str.len().median())}bp", flush=True)

    for lr in a.lrs:
        for seed in a.seeds:
            tag = f"{a.model}__{a.regime}__{a.mod}_{a.split}_{lv}__s{seed}__lr{lr:g}"
            if os.path.exists(f"{OUT}/{tag}.json"): print(f"skip {tag}", flush=True); continue
            torch.manual_seed(seed); np.random.seed(seed)
            m = Wrap(mid, how, hid, ncls, a.regime).to(a.device)
            npar = sum(p.numel() for p in m.parameters())
            tp = [p for p in m.parameters() if p.requires_grad]; ntp = sum(p.numel() for p in tp)
            print(f"  [{a.model}/{a.regime}] params={npar/1e6:.2f}M trainable={ntp/1e6:.3f}M ({100*ntp/npar:.2f}%)", flush=True)
            opt = torch.optim.AdamW(tp, lr=lr, weight_decay=0.01)
            steps = max(1, a.epochs * (len(Itr) // a.bs))
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
            sc = torch.amp.GradScaler("cuda")

            def pred(I, M):
                m.eval(); o = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(0, len(I), a.bs * 2):
                        mm = M[i:i + a.bs * 2].to(a.device) if M is not None else None
                        o.append(m(I[i:i + a.bs * 2].to(a.device), mm).float().argmax(-1).cpu().numpy())
                m.train(); return np.concatenate(o)

            best = (-1, 0, None); traj = []; noimp = 0
            for ep in range(1, a.epochs + 1):
                perm = torch.randperm(len(Itr)); t0 = time.time()
                for i in range(0, len(perm) - a.bs + 1, a.bs):
                    b = perm[i:i + a.bs]
                    mm = Mtr[b].to(a.device) if Mtr is not None else None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(m(Itr[b].to(a.device), mm), ytr[b].to(a.device))
                    opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt)
                    nn.utils.clip_grad_norm_(tp, 1.0); sc.step(opt); sc.update(); sch.step()
                f_dv = f1_score(ydv, pred(Idv, Mdv), average="macro", zero_division=0)
                pt = pred(Ite, Mte); f_te = f1_score(yte, pt, average="macro", zero_division=0)
                traj.append(dict(epoch=ep, dev_f1=round(f_dv, 4), test_f1=round(f_te, 4)))
                print(f"  [{tag}] ep{ep:3d} dev_f1={f_dv:.4f} test_f1={f_te:.4f} ({time.time()-t0:.0f}s)", flush=True)
                if f_dv > best[0] + 5e-4:
                    best = (f_dv, ep, pt.copy()); noimp = 0
                else:
                    noimp += 1
                    if noimp >= 5: print("  early stop", flush=True); break
            p = best[2]
            mm = metrics(yte, p)
            res = dict(model=a.model, regime=a.regime, mod=a.mod, split=a.split, level=lv,
                       seed=seed, lr=lr, n_classes=ncls, n_params=npar, n_trainable=ntp,
                       trainable_frac=round(ntp / npar, 5), maxlen_tok=ml,
                       effective_context_bp=eff_ctx_bp,
                       best_epoch=best[1], epochs_run=len(traj), dev_macro_f1=round(best[0], 4),
                       **mm, n_train=len(tr), trajectory=traj)
            json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
            print(f"  RESULT {tag}: test_macro_f1={mm['macro_f1']:.4f} mcc={mm['mcc']:.4f}", flush=True)
            del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
