"""Splice FINE-TUNED arm -- separates two explanations for the frozen-probe shortfall.

The frozen probe on splice reaches only 0.32-0.59 MCC while published FM numbers are
0.97-0.98. Two candidate explanations:
  (A) published splice numbers are FINE-TUNED, and frozen probing simply cannot reach them
  (B) our extraction/pipeline is broken
This runs full fine-tuning of the same checkpoints on the same splits. If (A), FT lands near
0.97; if (B), FT also underperforms and the pipeline is the problem.

Comparator: the dev-selected supervised baseline from capacity_sweep.py (0.9527-0.9637 MCC).
Selection on dev only; test scored at the best-dev epoch.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, torch.nn as nn
import paths as P
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import matthews_corrcoef
import logging, transformers
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
import warnings; warnings.filterwarnings("ignore")

OUT = P.sub("splice_finetune"); NT_DIR = os.environ.get("VB_NT_DIR", "/data/nvidia/data/ntv3")
MODELS = {
 "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 1024),
 "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t", 512),
 "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", 1024),
}

class EncoderWithHead(nn.Module):
    """AutoModel encoder + mean-pool + linear head.

    Used where AutoModelForSequenceClassification would silently re-initialise pretrained
    weights (GENA-LM). Loading via AutoModel is also the exact path the frozen probes use,
    so the frozen-vs-FT regime comparison shares an identical encoder.
    """
    def __init__(self, mid, ncls):
        super().__init__()
        self.enc = AutoModel.from_pretrained(mid, trust_remote_code=True)
        hid = getattr(self.enc.config, "hidden_size", None) or self.enc.config.d_model
        self.head = nn.Linear(hid, ncls)

    def forward(self, input_ids, attention_mask=None, labels=None):
        o = self.enc(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o.last_hidden_state
        if attention_mask is None:
            pooled = h.mean(1)
        else:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        logits = self.head(pooled)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return type("O", (), out)


def assert_no_fresh_encoder_weights(mid, cls_name):
    """Fail loudly if the chosen load path re-initialises pretrained tensors."""
    import io, logging, re, transformers as T
    buf = io.StringIO(); h = logging.StreamHandler(buf)
    lg = logging.getLogger("transformers.modeling_utils")
    prev = lg.level; lg.addHandler(h); lg.setLevel(logging.WARNING)
    T.logging.set_verbosity_warning()
    try:
        {"seqcls": AutoModelForSequenceClassification, "automodel": AutoModel}[cls_name]\
            .from_pretrained(mid, trust_remote_code=True)
    finally:
        lg.removeHandler(h); lg.setLevel(prev); T.logging.set_verbosity_error()
    txt = buf.getvalue()
    bad = re.findall(r"newly initialized: \[(.*?)\]", txt, re.S)
    enc_bad = [x for x in re.findall(r"'([^']+)'", bad[0])] if bad else []
    # the task head MUST be freshly initialised -- that is not a bug. Different checkpoints
    # name it differently (BERT: classifier, HyenaDNA: score, others: head / cls.*).
    HEAD_NAMES = ("classifier", "head", "score", "cls.")
    enc_bad = [k for k in enc_bad if not any(h in k for h in HEAD_NAMES)]
    if enc_bad:
        raise SystemExit(f"ABORT: {cls_name} re-initialises {len(enc_bad)} pretrained tensors "
                         f"for {mid} (e.g. {enc_bad[:3]}). Use the AutoModel+head path instead.")
    print(f"  weight-load check OK ({cls_name}): no pretrained tensor re-initialised", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--load_path", default="auto", choices=["auto","seqcls","automodel"])
    ap.add_argument("--min_epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    trall = pd.read_parquet(f"{NT_DIR}/{a.task}/train.parquet")
    te = pd.read_parquet(f"{NT_DIR}/{a.task}/test.parquet")
    rng = np.random.default_rng(0); m = np.zeros(len(trall), bool)
    m[rng.permutation(len(trall))[:int(.15*len(trall))]] = True
    tr, dv = trall[~m].reset_index(drop=True), trall[m].reset_index(drop=True)
    ncls = int(pd.concat([tr,dv,te]).label.nunique())
    mid, ml = MODELS[a.model]
    print(f"{a.task} / {a.model}: cls={ncls} train={len(tr)} dev={len(dv)} test={len(te)} maxlen={ml}", flush=True)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    def tk(seqs):
        e = tok(list(seqs), padding="longest", truncation=True, max_length=ml, return_tensors="pt")
        return e["input_ids"], e.get("attention_mask")
    Xtr, Mtr = tk(tr.sequence); Xdv, Mdv = tk(dv.sequence); Xte, Mte = tk(te.sequence)
    print(f"  tokenized to {Xtr.shape[1]} tokens (model max {ml})", flush=True)
    ytr = torch.tensor(tr.label.values, dtype=torch.long)
    runs = []
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        if a.load_path == "auto":
            path = "automodel" if a.model == "gena_lm" else "seqcls"
        else:
            path = a.load_path
        if seed == a.seeds[0]:
            assert_no_fresh_encoder_weights(mid, path)
        if path == "automodel":
            model = EncoderWithHead(mid, ncls).to(a.device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                mid, num_labels=ncls, trust_remote_code=True).to(a.device)
        ntr_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        # constant LR with no warmup collapsed every run to the majority class (dev MCC 0.0000)
        steps_per_ep = max(1, (len(Xtr) // a.bs) // a.accum)
        from transformers import get_linear_schedule_with_warmup
        total_steps = steps_per_ep * a.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
        def ev(X, M):
            model.eval(); pr = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(X), 32):
                    kw = {"attention_mask": M[i:i+32].to(a.device)} if M is not None else {}
                    pr.append(model(X[i:i+32].to(a.device), **kw).logits.float().argmax(-1).cpu().numpy())
            model.train(); return np.concatenate(pr)
        best = (-1, 0, None); noimp = 0
        for ep in range(1, a.epochs+1):
            perm = torch.randperm(len(Xtr)); opt.zero_grad(); t0 = time.time()
            for k, i in enumerate(range(0, len(perm)-a.bs+1, a.bs)):
                b = perm[i:i+a.bs]
                kw = {"attention_mask": Mtr[b].to(a.device)} if Mtr is not None else {}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(Xtr[b].to(a.device), **kw, labels=ytr[b].to(a.device)).loss / a.accum
                loss.backward()
                if (k+1) % a.accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad()
            d = matthews_corrcoef(dv.label.values, ev(Xdv, Mdv))
            print(f"  s{seed} ep{ep} dev_mcc={d:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if d > best[0] + 1e-4:
                best = (d, ep, matthews_corrcoef(te.label.values, ev(Xte, Mte))); noimp = 0
            else:
                noimp += 1
                # a degenerate first epoch is not convergence -- never stop before min_epochs
                if ep >= a.min_epochs and noimp >= a.patience: break
        print(f"  s{seed} BEST dev={best[0]:.4f} @ep{best[1]} -> TEST MCC={best[2]:.4f}", flush=True)
        runs.append(dict(seed=seed, dev=round(best[0],4), test=round(best[2],4), best_epoch=best[1],
                         trainable_params=ntr_p))
        del model; torch.cuda.empty_cache()
    collapsed = all(abs(r['dev']) < 1e-6 for r in runs)
    res = dict(task=a.task, model=a.model, regime="full_finetune", n_classes=ncls, max_len=ml,
               n_train=len(tr), n_test=len(te), lr=a.lr, warmup="linear 10%", load_path=a.load_path, runs=runs,
               collapsed_to_majority_class=collapsed,
               test_mean=round(float(np.mean([r['test'] for r in runs])),4))
    if collapsed:
        print("  WARNING: collapsed to majority class -- NOT a usable FT capability estimate", flush=True)
    json.dump(res, open(f"{OUT}/{a.task}__{a.model}__lr{a.lr:g}__fullft.json","w"), indent=2)
    print(f"  wrote {OUT}/{a.task}__{a.model}__lr{a.lr:g}__fullft.json", flush=True)

if __name__ == "__main__":
    main()
