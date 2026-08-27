"""Full fine-tuning + LR sweep for gLMs on GENEB tasks.

GENEB itself only reports a frozen linear probe on precomputed embeddings, and our own splice
measurement shows the frozen->FT gap (0.59 MCC) is larger than any gap between models. So a
probe-only number is evidence about the probe, not about capability. This adds the FT arm.

Protocol, matched to Cluster 2's fair-k-mer run so the numbers are comparable:
  - GENEB ships train.csv / test.csv only (no dev). Dev is a STRATIFIED 15% carve of train,
    seed 42 -- identical to scripts/geneb/fair_kmer_sentinel.py.
  - LR selected on dev; test scored once at the best-dev epoch.
  - Primary metric MCC (what GENEB reports); accuracy and macro-F1 also recorded.
  - Linear warmup 10% + linear decay: constant LR with no warmup collapses these models to the
    majority class (see splice_finetune.py).
  - assert_no_fresh_encoder_weights() aborts if a load path silently re-initialises pretrained
    tensors -- GENA-LM via AutoModelForSequenceClassification discards all 48 pretrained LayerNorms.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, torch.nn as nn
import paths as P
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModel,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import logging, transformers
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
import warnings; warnings.filterwarnings("ignore")

OUT = P.sub("geneb_finetune")
GENEB = os.environ.get("VB_GENEB_DIR", "/data/nvidia/geneb_data/tasks")
MODELS = {
 "nt_v2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 1024, 6),
 "gena_lm":    ("AIRI-Institute/gena-lm-bert-base-t2t", 512, 6),
 "hyenadna":   ("LongSafari/hyenadna-medium-160k-seqlen-hf", 8192, 1),
}
HEAD_NAMES = ("classifier", "head", "score", "cls.")


class EncoderWithHead(nn.Module):
    """AutoModel + masked mean-pool + linear head, for checkpoints where
    AutoModelForSequenceClassification would silently re-initialise pretrained weights."""
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


def assert_no_fresh_encoder_weights(mid, path):
    import io, re
    buf = io.StringIO(); h = logging.StreamHandler(buf)
    lg = logging.getLogger("transformers.modeling_utils"); prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.WARNING); transformers.logging.set_verbosity_warning()
    try:
        {"seqcls": AutoModelForSequenceClassification, "automodel": AutoModel}[path]\
            .from_pretrained(mid, trust_remote_code=True)
    finally:
        lg.removeHandler(h); lg.setLevel(prev); transformers.logging.set_verbosity_error()
    bad = re.findall(r"newly initialized: \[(.*?)\]", buf.getvalue(), re.S)
    names = [k for k in (re.findall(r"'([^']+)'", bad[0]) if bad else [])
             if not any(x in k for x in HEAD_NAMES)]
    if names:
        raise SystemExit(f"ABORT: {path} re-initialises {len(names)} pretrained tensors for {mid} "
                         f"(e.g. {names[:3]}). Use --load_path automodel.")
    print(f"  weight-load check OK ({path})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--min_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--cap_train", type=int, default=0, help="0 = all rows")
    ap.add_argument("--load_path", default="auto", choices=["auto","seqcls","automodel"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)

    tr_all = pd.read_csv(f"{GENEB}/{a.task}/train.csv")
    te = pd.read_csv(f"{GENEB}/{a.task}/test.csv")
    if a.cap_train and len(tr_all) > a.cap_train:
        tr_all = tr_all.sample(n=a.cap_train, random_state=0).reset_index(drop=True)
    # dev carve IDENTICAL to fair_kmer_sentinel.py: stratified 15%, seed 42
    idx = np.arange(len(tr_all))
    i_tr, i_dv = train_test_split(idx, test_size=0.15, stratify=tr_all.label.values, random_state=42)
    tr, dv = tr_all.iloc[i_tr].reset_index(drop=True), tr_all.iloc[i_dv].reset_index(drop=True)
    ncls = int(pd.concat([tr_all, te]).label.nunique())
    mid, model_max, bp_per_tok = MODELS[a.model]
    need = int(pd.concat([tr_all, te]).text.str.len().max())
    ml = min(model_max, max(16, need // bp_per_tok + 4))
    print(f"{a.task} / {a.model} lr={a.lr:g}: cls={ncls} train={len(tr)} dev={len(dv)} test={len(te)} "
          f"maxbp={need} -> maxlen={ml} tok", flush=True)

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    def tk(seqs):
        e = tok(list(seqs), padding="longest", truncation=True, max_length=ml, return_tensors="pt")
        return e["input_ids"], e.get("attention_mask")
    Xtr, Mtr = tk(tr.text); Xdv, Mdv = tk(dv.text); Xte, Mte = tk(te.text)
    print(f"  tokenized to {Xtr.shape[1]} tokens", flush=True)
    ytr = torch.tensor(tr.label.values, dtype=torch.long)

    path = ("automodel" if a.model == "gena_lm" else "seqcls") if a.load_path == "auto" else a.load_path
    runs = []
    for si, seed in enumerate(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        if si == 0: assert_no_fresh_encoder_weights(mid, path)
        model = (EncoderWithHead(mid, ncls) if path == "automodel"
                 else AutoModelForSequenceClassification.from_pretrained(
                     mid, num_labels=ncls, trust_remote_code=True)).to(a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        spe = max(1, (len(Xtr) // a.bs) // a.accum); total = spe * a.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        def ev(X, M):
            model.eval(); pr = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(X), 32):
                    kw = {"attention_mask": M[i:i+32].to(a.device)} if M is not None else {}
                    pr.append(model(X[i:i+32].to(a.device), **kw).logits.float().argmax(-1).cpu().numpy())
            model.train(); return np.concatenate(pr)
        best = (-2.0, 0, None); noimp = 0
        for ep in range(1, a.epochs + 1):
            perm = torch.randperm(len(Xtr)); opt.zero_grad(); t0 = time.time()
            for k, i in enumerate(range(0, len(perm) - a.bs + 1, a.bs)):
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
                p = ev(Xte, Mte); best = (d, ep, p); noimp = 0
            else:
                noimp += 1
                if ep >= a.min_epochs and noimp >= a.patience: break
        p = best[2]
        r = dict(seed=seed, dev_mcc=round(best[0], 4), best_epoch=best[1],
                 test_mcc=round(float(matthews_corrcoef(te.label.values, p)), 4),
                 test_acc=round(float(accuracy_score(te.label.values, p)), 4),
                 test_macro_f1=round(float(f1_score(te.label.values, p, average="macro", zero_division=0)), 4),
                 n_pred_classes=int(len(np.unique(p))))
        print(f"  s{seed} BEST dev={r['dev_mcc']:.4f} @ep{r['best_epoch']} -> TEST MCC={r['test_mcc']:.4f}", flush=True)
        runs.append(r)
        np.savez_compressed(f"{OUT}/{a.task}__{a.model}__lr{a.lr:g}__s{seed}__preds.npz",
                            y_true=te.label.values, y_pred=p)
        del model; torch.cuda.empty_cache()

    collapsed = all(abs(r["dev_mcc"]) < 1e-6 for r in runs)
    res = dict(task=a.task, model=a.model, regime="full_finetune", lr=a.lr, load_path=path,
               n_classes=ncls, max_len_tok=ml, n_train=len(tr), n_dev=len(dv), n_test=len(te),
               cap_train=a.cap_train, warmup="linear 10%", dev_carve="stratified 15% seed 42",
               collapsed_to_majority_class=collapsed, runs=runs,
               test_mcc_mean=round(float(np.mean([r["test_mcc"] for r in runs])), 4),
               test_mcc_sd=round(float(np.std([r["test_mcc"] for r in runs], ddof=1)), 4) if len(runs) > 1 else None)
    if collapsed: print("  WARNING: collapsed to majority class -- NOT a capability estimate", flush=True)
    json.dump(res, open(f"{OUT}/{a.task}__{a.model}__lr{a.lr:g}__fullft.json", "w"), indent=2)
    print(f"  wrote {a.task}__{a.model}__lr{a.lr:g}__fullft.json", flush=True)

if __name__ == "__main__":
    main()
