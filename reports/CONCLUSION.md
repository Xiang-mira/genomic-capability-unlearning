# Conclusion — Genomic Capability Unlearning Evaluation

**Date:** 2026-07-20
**Model:** Evo-1-8k-base (1.1B StripedHyena GLM)
**Scope:** Whether the model has a genuine, generalizable dangerous capability (human-virus pathogenicity / host-tropism / transmissibility discrimination) that exceeds trivial k-mer composition shortcuts — the precondition for meaningful unlearning.

---

## Headline finding

**On the tasks examined, the model's apparent capability above the k-mer baseline is largely a composition/genus-overlap artifact, not generalizable capability.** When evaluation removes composition-neighbourhood overlap between train and test, the model's edge over a bag-of-k-mers classifier collapses to zero.

This means: for the frozen representation, **there is little generalizable dangerous capability to unlearn** — a plain k-mer logistic regression reproduces the model's performance once you evaluate honestly. Representation-space unlearning (RMU / probe-guided) has essentially nothing task-specific to erase beyond composition statistics.

---

## Evidence

### 1. Direct, decisive — bvbrc_cov frozen-probe under composition-disjoint splits
(`reports/model_vs_kmer_cluster_disjoint.csv`, GPU-free, cached layer-0 features, 5,632 seqs)

| Split | k-mer AUROC | model probe (L0) | model excess (model − k-mer) |
|-------|-------------|------------------|------------------------------|
| Random | 0.751 | 0.906 | **+0.155** |
| Composition-cluster-disjoint | 0.717 | 0.701 | **−0.016** |
| Drop (random→disjoint) | −0.035 | **−0.206** | — |

The +0.155 excess on random splits **collapses to −0.016** when whole composition clusters are held out. The model representation drops 6× more than k-mer (−0.206 vs −0.035): it encodes composition-neighbourhood identity, not generalizable pathogenicity.

### 2. Corroborating — ViroBench host prediction across split strategies
(`reports/split_diagnostics.csv`, from glm-locking exp3)

| Split | pretrained residual vs k-mer | unlocked-FT residual |
|-------|------------------------------|----------------------|
| Random CV | −0.022 (neg) | −0.123 (neg) |
| LOFO (family-disjoint) | −0.097 (neg) | −0.106 (neg) |
| Temporal | −0.015 (ns) | −0.244 (neg) |

The model **never beats k-mer** on host prediction under any split, and — critically — **full fine-tuning makes generalization WORSE** (−0.244 temporal), i.e. it overfits training-family composition. This is the weight-space analogue of the frozen-probe collapse in (1).

### 3. Shortcut ceilings are high and were being under-measured
(`reports/shortcut_audit.csv`)

Full-training-set k-mer(3–6) logistic regression: bvbrc_cov **0.948**, cini **0.850**, host_tropism **0.905**; HVUE aggregates ≥0.87 (still rising at C=10, so lower bounds). Nearest-neighbour on random splits reaches 0.88–0.96 — the composition-retrieval ceiling. A sample-capped audit had spuriously made NN look dominant; corrected on full data, kmer3_6/raw+kmer is the true ceiling.

### 4. Every candidate is Category B, and all frozen probes sit at/below the k-mer ceiling

| Task | frozen probe | k-mer ceiling | probe − k-mer | Category |
|------|-------------|---------------|---------------|----------|
| bvbrc_cov | 0.939 | 0.948 | −0.009 | B |
| cini | 0.798 | 0.850 | −0.052 | B |
| host_tropism | 0.909 | 0.905 | +0.004 | boundary |

RMU (early-layer activation misdirection) is structurally the wrong tool: there is no probe-accessible capability above composition to misdirect.

---

## What this means for the unlearning program

1. **Frozen-representation unlearning (RMU/probe) is not warranted** on these tasks — the target capability is not present in the frozen representation beyond composition statistics that a k-mer model already captures.

2. **The apparent "excess" that motivated unlearning is a random-split artifact.** Any future capability claim (or unlearning success claim) MUST be measured under a genus/composition-disjoint split. Random-split AUROC gaps are not evidence of capability.

3. **HVUE aggregate excess is unconfirmed.** On random splits, unlocked full fine-tuning shows +0.098 (Pathogenicity) / +0.055 (Host_Tropism) / +0.037 (Transmissibility) over the (underestimated) k-mer ceiling. By analogy to ViroBench unlocked-FT — which got *worse* under disjoint splits — these are expected to shrink or reverse under honest evaluation. **Not yet tested for the HVUE tasks.**

4. **The 5000-step LoRA adversary run was broken** (6/7 retain tasks trained degenerately, MCC=0; lr=1e-4, no warmup, patience=3) and was terminated. It produced no usable ceilings. glm-locking's own LoRA pipeline (stable, 0.90–0.97 on random splits) is the reference for weight-space numbers.

---

## Open question (deferred)

The one thing not settled: does the **weight-space** capability that full fine-tuning unlocks *also* collapse under a composition/genus-disjoint split for the HVUE pathogenicity tasks specifically? ViroBench host says yes (unlocked-FT worsens under disjoint splits); the frozen-probe result for bvbrc_cov says yes. To close it definitively would require a *stable* LoRA fine-tune (fixed config: warmup + higher patience) evaluated on the composition-cluster-disjoint splits already built here. GPUs are now free for this if desired.

**Working conclusion pending that test:** the dangerous-capability framing is not supported by generalizable evidence — the model's viral-pathogenicity discrimination is, to the resolution we can measure, k-mer composition retrieval that a simple baseline reproduces. Unlearning it would remove nothing a k-mer classifier does not already provide.

---

## Artifacts produced

| File | Content |
|------|---------|
| `reports/task_inventory.md` | 11 tasks: stats, labels, shortcut risk, classification |
| `reports/shortcut_audit.csv` | Full-train k-mer/NN baselines (authoritative) |
| `reports/shortcut_audit_5000cap.csv` | Sample-capped audit (deprecated; kept for provenance) |
| `reports/base_vs_shortcut.csv` | Model vs shortcut, per task, with excess + validity flags |
| `reports/split_diagnostics.csv` | ViroBench LOFO/temporal + composition-cluster holdout results |
| `reports/composition_cluster_holdout.csv` | k-mer/NN degradation under cluster-disjoint splits |
| `reports/model_vs_kmer_cluster_disjoint.csv` | **Decisive**: bvbrc_cov model excess collapse |
| `reports/recommended_forget_tasks.md` | Classification + why no PRIMARY_FORGET confirmed |
| `reports/recommended_retain_tasks.md` | Retain suite + tolerances |
| `reports/unlearning_eval_protocol.md` | Full protocol, criteria, all findings |
| `phase2/composition_cluster_holdout.py`, `phase2/model_vs_kmer_cluster_disjoint.py` | Reusable disjoint-split diagnostics |
