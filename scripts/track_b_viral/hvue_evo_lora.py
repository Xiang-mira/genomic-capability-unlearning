"""
Evo-1-8k LoRA on the v2 splits, with the three protocol flaws fixed.

Faithful to lora_confound_v2.py (same LoRA rank/alpha/targets, same iv slice, same
frozen-Youden threshold) except:
  1. ev evaluated at EVERY checkpoint (on a fixed subsample for the trajectory; the
     full ev partition once at the best-iv step for the headline number).
  2. --lr extended beyond 1e-4, which was the ceiling of the original grid and was
     selected in every single cell.
  3. --max_steps raised with a real min_delta/patience so runs are not truncated by
     the cap. All 18 original runs ended at max_steps; 0 early-stopped.

Writes to scratchpad/multimodel/evo_results — the original model_results_v2/ is untouched.
"""
import os, sys, json, time, argparse, itertools, gc, math, warnings
import numpy as np, pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")
ROOT = P.LOCK_ROOT; sys.path.insert(0, ROOT)
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from src.utils import load_evo_model, get_amp_settings
from src.lora import inject_lora
from scripts.hvue_lora_finetune import (
    HVUEDataset, collate_pad, ClassificationHead, find_norm, forward_with_hook,
    LORA_RANK, LORA_ALPHA, LORA_TARGETS, BATCH_SIZE, GRAD_ACCUM)

SPL = P.SPLITS_V2
OUT = P.sub("hvue_evo")
KMER = {("Host_Tropism","random"):0.9213, ("Host_Tropism","cluster_disjoint"):0.8034,
        ("Pathogenecity","random"):0.9685, ("Pathogenecity","cluster_disjoint"):0.8044,
        ("Transmissibility","random"):0.9238, ("Transmissibility","cluster_disjoint"):0.7395}
WARMUP = 100


def eval_logits(model, head, loader, norm, device, amp):
    model.eval(); head.eval(); lg, ys = [], []
    with torch.no_grad():
        for ids_v, lab in loader:
            ids_v = ids_v.to(device)
            with torch.autocast(device_type="cuda", dtype=amp):
                h = forward_with_hook(model, ids_v, norm); lo = head(h.float())
            lg.append(lo.cpu().float().numpy()); ys.append(lab.numpy())
    model.train(); head.train()
    return np.concatenate(lg), np.concatenate(ys).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Host_Tropism"); ap.add_argument("--split", default="cluster_disjoint")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=5e-4)
    ap.add_argument("--ev_sub", type=int, default=3000)
    a = ap.parse_args()
    amp, _ = get_amp_settings(); dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(f"{SPL}/{a.task}__{a.split}.parquet")
    tr_all = df[df.partition == "train"].reset_index(drop=True)
    ev_df = df[df.partition == "val"].reset_index(drop=True)
    sub = ev_df.sample(min(a.ev_sub, len(ev_df)), random_state=0).reset_index(drop=True)
    km = KMER[(a.task, a.split)]

    for lr in a.lrs:
        for seed in a.seeds:
            tag = f"{a.task}__{a.split}__s{seed}__lr{lr:g}"
            if os.path.exists(f"{OUT}/{tag}.json"):
                print(f"skip {tag}", flush=True); continue
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(seed); ivm = np.zeros(len(tr_all), bool)
            for c in tr_all.label.unique():
                idx = np.where(tr_all.label.values == c)[0]; rng.shuffle(idx)
                ivm[idx[:int(0.15 * len(idx))]] = True
            iv_df = tr_all[ivm].reset_index(drop=True); tr_df = tr_all[~ivm].reset_index(drop=True)
            model, tok = load_evo_model("evo-1-8k-base", dev)
            for p in model.parameters(): p.requires_grad = False
            inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, target_substrings=LORA_TARGETS)
            head = ClassificationHead(hidden=4096).to(dev).to(amp)
            norm = find_norm(model); model.train(); head.train()
            trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
            opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
            def lr_lambda(s):
                tot = max(a.max_steps // GRAD_ACCUM, 1); w = WARMUP // GRAD_ACCUM
                if s < w: return (s + 1) / max(w, 1)
                return 0.5 * (1 + math.cos(math.pi * min((s - w) / max(tot - w, 1), 1.0)))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            mk = lambda d, b: DataLoader(HVUEDataset(d, tok), batch_size=b, shuffle=False, collate_fn=collate_pad)
            tr_loader = DataLoader(HVUEDataset(tr_df, tok), batch_size=BATCH_SIZE, shuffle=True,
                                   collate_fn=collate_pad, drop_last=True)
            iv_loader, sub_loader, ev_loader = mk(iv_df, BATCH_SIZE*4), mk(sub, BATCH_SIZE*4), mk(ev_df, BATCH_SIZE*4)
            tr_iter = itertools.cycle(tr_loader)
            best = dict(iv=0.0, step=0, state=None, head=None, iv_preds=None, iv_y=None)
            traj = []; noimp = 0; opt.zero_grad(); t0 = time.time()
            for step in range(1, a.max_steps + 1):
                ids, lab = next(tr_iter)
                with torch.autocast(device_type="cuda", dtype=amp):
                    h = forward_with_hook(model, ids.to(dev), norm); logit = head(h.float())
                    loss = F.binary_cross_entropy_with_logits(logit, lab.to(dev)) / GRAD_ACCUM
                loss.backward()
                if step % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad()
                if step % a.val_every == 0 or step == a.max_steps:
                    ivp, ivy = eval_logits(model, head, iv_loader, norm, dev, amp)
                    sp, sy = eval_logits(model, head, sub_loader, norm, dev, amp)
                    aiv, aev = float(roc_auc_score(ivy, ivp)), float(roc_auc_score(sy, sp))
                    traj.append(dict(step=step, iv_auroc=round(aiv, 4), ev_sub_auroc=round(aev, 4)))
                    print(f"  [{tag}] step{step:5d} iv={aiv:.4f} ev_sub={aev:.4f} ({time.time()-t0:.0f}s)", flush=True)
                    if aiv > best["iv"] + a.min_delta:
                        best.update(iv=aiv, step=step, iv_preds=ivp.copy(), iv_y=ivy.copy(),
                                    state={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                                    head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                        noimp = 0
                    else:
                        noimp += 1
                        if noimp >= a.patience:
                            print(f"  early stop @ {step}", flush=True); break
            for n, p in model.named_parameters():
                if p.requires_grad: p.data.copy_(best["state"][n].to(dev))
            head.load_state_dict({k: v.to(dev) for k, v in best["head"].items()})
            evp, evy = eval_logits(model, head, ev_loader, norm, dev, amp)
            fpr, tpr, th = roc_curve(best["iv_y"], best["iv_preds"]); thr = float(th[np.argmax(tpr - fpr)])
            ev_a = float(roc_auc_score(evy, evp))
            orc = max(t["ev_sub_auroc"] for t in traj); mn = min(t["ev_sub_auroc"] for t in traj)
            res = dict(model="evo1_8k_lora", task=a.task, split=a.split, seed=seed, lr=lr,
                       best_step=best["step"], steps_run=traj[-1]["step"], n_checkpoints=len(traj),
                       early_stopped=bool(traj[-1]["step"] < a.max_steps),
                       iv_auroc=round(best["iv"], 4), ev_auroc=round(ev_a, 4),
                       ev_mcc=round(float(matthews_corrcoef(evy, (evp >= thr).astype(int))), 4),
                       kmer_auroc=km, excess=round(ev_a - km, 4),
                       ev_sub_oracle=round(orc, 4), ev_sub_min=round(mn, 4), ev_sub_range=round(orc - mn, 4),
                       n_train=len(tr_df), n_iv=len(iv_df), n_ev=len(ev_df), trajectory=traj)
            json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=2)
            print(f"  RESULT {tag}: ev={ev_a:.4f} kmer={km:.4f} excess={ev_a-km:+.4f} "
                  f"early_stopped={res['early_stopped']} ev_sub range over {len(traj)} ckpts={res['ev_sub_range']:.4f}", flush=True)
            del model, head; gc.collect(); torch.cuda.empty_cache(); time.sleep(2)


if __name__ == "__main__":
    main()
