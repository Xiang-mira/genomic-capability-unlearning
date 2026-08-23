"""
gLMs on the two GUE viral tasks the repo dropped. Multiclass.

  virus_covid       9 SARS-CoV-2 variants, 73,335/9,166/9,168, ~1kb
  virus_species_40  25 virus species,       4,000/500/500,     ~5kb

Same discipline as glm_finetune.py: regime in {probe,lora,full}, model selected on the
GUE dev split, macro-F1 reported on the GUE test split, test evaluated EVERY epoch so
the trajectory is visible. Reference baselines are in gue_results/*_baselines.json.

GUE's own metric for the virus tasks is macro-F1.
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")
D = P.GUE_DIR
OUT = P.sub("gue_glm")
MODELS = {
  "hyenadna":  ("LongSafari/hyenadna-medium-160k-seqlen-hf", "AutoModel", 256),
  "gena_lm":   ("AIRI-Institute/gena-lm-bert-base-t2t",      "AutoModel", 768),
  "nt_v2_500m":("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species","AutoModelForMaskedLM",1024),
}


class Wrap(nn.Module):
    def __init__(self, mid, how, hid, ncls, regime="full", lora_rank=16, random_init=False):
        super().__init__()
        cls = {"AutoModel": AutoModel, "AutoModelForMaskedLM": AutoModelForMaskedLM}[how]
        if random_init:
            from transformers import AutoConfig
            self.body = cls.from_config(AutoConfig.from_pretrained(mid, trust_remote_code=True),
                                        trust_remote_code=True)
        else:
            self.body = cls.from_pretrained(mid, trust_remote_code=True)
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
            m = mask.unsqueeze(-1).to(h.dtype); pooled = (h*m).sum(1)/m.sum(1).clamp(min=1)
        return self.head(pooled.float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task", required=True, choices=["virus_covid", "virus_species_40"])
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--regime", default="full", choices=["probe", "lora", "full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-5])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=None); ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--random_init", action="store_true"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    mid, how, hid = MODELS[a.model]
    ml = a.maxlen or {"hyenadna": 1024 if a.task == "virus_covid" else 5120,
                      "gena_lm": 256 if a.task == "virus_covid" else 512,
                      "nt_v2_500m": 176 if a.task == "virus_covid" else 860}[a.model]
    tr = pd.read_csv(f"{D}/{a.task}__train.csv"); dv = pd.read_csv(f"{D}/{a.task}__dev.csv")
    te = pd.read_csv(a.test_csv if a.test_csv else f"{D}/{a.task}__test.csv")
    if a.test_csv:
        print(f"  TEST SET OVERRIDE: {a.test_csv} (n={len(te)})", flush=True)
    if len(tr) > a.cap: tr = tr.sample(a.cap, random_state=0).reset_index(drop=True)
    ncls = int(pd.concat([tr, dv, te]).label.nunique())
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    def tk(s):
        e = tok(list(s), return_tensors="pt", padding="max_length", truncation=True, max_length=ml)
        return e["input_ids"], e.get("attention_mask")
    Itr, Mtr = tk(tr.sequence); Idv, Mdv = tk(dv.sequence); Ite, Mte = tk(te.sequence)
    ytr = torch.tensor(tr.label.values); ydv = dv.label.values; yte = te.label.values
    print(f"{a.model}/{a.task}/{a.regime}: n_tr={len(tr)} classes={ncls} maxlen={ml} tok_len={Itr.shape[1]}", flush=True)

    for lr in a.lrs:
        for seed in a.seeds:
            tag = f"{a.model}{'_randinit' if a.random_init else ''}__{a.regime}__{a.task}__s{seed}__lr{lr:g}"
            if os.path.exists(f"{OUT}/{tag}.json"): print(f"skip {tag}", flush=True); continue
            torch.manual_seed(seed); np.random.seed(seed)
            m = Wrap(mid, how, hid, ncls, a.regime, random_init=a.random_init).to(a.device)
            npar = sum(p.numel() for p in m.parameters())
            tp = [p for p in m.parameters() if p.requires_grad]; ntp = sum(p.numel() for p in tp)
            print(f"  [{a.model}/{a.regime}] params={npar/1e6:.2f}M trainable={ntp/1e6:.3f}M ({100*ntp/npar:.2f}%)", flush=True)
            opt = torch.optim.AdamW(tp, lr=lr, weight_decay=0.01)
            steps = max(1, a.epochs*(len(Itr)//a.bs))
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
            sc = torch.amp.GradScaler("cuda")
            def pred(I, M):
                m.eval(); o = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(0, len(I), a.bs*2):
                        mm = M[i:i+a.bs*2].to(a.device) if M is not None else None
                        o.append(m(I[i:i+a.bs*2].to(a.device), mm).float().argmax(-1).cpu().numpy())
                m.train(); return np.concatenate(o)
            best = (-1, 0, None); traj = []; noimp = 0
            for ep in range(1, a.epochs+1):
                perm = torch.randperm(len(Itr)); t0 = time.time()
                for i in range(0, len(perm)-a.bs+1, a.bs):
                    b = perm[i:i+a.bs]
                    mm = Mtr[b].to(a.device) if Mtr is not None else None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(m(Itr[b].to(a.device), mm), ytr[b].to(a.device))
                    opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt)
                    nn.utils.clip_grad_norm_(tp, 1.0); sc.step(opt); sc.update(); sch.step()
                f_dv = f1_score(ydv, pred(Idv, Mdv), average="macro", zero_division=0)
                pt = pred(Ite, Mte); f_te = f1_score(yte, pt, average="macro", zero_division=0)
                traj.append(dict(epoch=ep, dev_f1=round(f_dv, 4), test_f1=round(f_te, 4)))
                print(f"  [{tag}] ep{ep:3d} dev_f1={f_dv:.4f} test_f1={f_te:.4f} ({time.time()-t0:.0f}s)", flush=True)
                if f_dv > best[0]+5e-4:
                    best = (f_dv, ep, pt.copy()); noimp = 0
                else:
                    noimp += 1
                    if noimp >= 5: print("  early stop", flush=True); break
            p = best[2]
            res = dict(model=a.model+('_randinit' if a.random_init else ''), regime=a.regime, task=a.task,
                       seed=seed, lr=lr, n_classes=ncls, n_params=npar, n_trainable=ntp,
                       trainable_frac=round(ntp/npar, 5), maxlen=ml, best_epoch=best[1], epochs_run=len(traj),
                       dev_macro_f1=round(best[0], 4),
                       test_macro_f1=round(float(f1_score(yte, p, average="macro", zero_division=0)), 4),
                       test_accuracy=round(float(accuracy_score(yte, p)), 4),
                       test_mcc=round(float(matthews_corrcoef(yte, p)), 4),
                       test_f1_oracle=round(max(t["test_f1"] for t in traj), 4),
                       test_f1_range=round(max(t["test_f1"] for t in traj)-min(t["test_f1"] for t in traj), 4),
                       n_train=len(tr), trajectory=traj)
            json.dump(res, open(f"{OUT}/{tag}" + ("__dedup" if a.test_csv else "") + f".json", "w"), indent=2)
            print(f"  RESULT {tag}: test_macro_f1={res['test_macro_f1']:.4f} mcc={res['test_mcc']:.4f} "
                  f"oracle={res['test_f1_oracle']:.4f} range={res['test_f1_range']:.4f}", flush=True)
            del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
