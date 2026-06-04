# Phase 2 Unlearning 结果分析

## 实验设置

**目标**：在 Evo-1-8k-base 上实现 human-virus-relevant capabilities 的靶向遗忘，比较不同 unlearning 方法和 layer condition 的效果。

**训练集构造**：
- **Forget optimization set**：human-tropic viral 序列（label=1，3800 条训练样本）
- **Retain optimization set**：non-human-tropic viral 序列（label=0，3814 条训练样本）

**主评估升级**：不再把 frozen host-tropism probe AUROC + forget/retain perplexity 作为 selective unlearning 的唯一证据。Phase 2 的主评估改为外部 HVUE + GUE benchmarks。当前工作区状态已由 `data/phase2/experiment_audit.json` 审计：vGUE 任务尚未落地到本地数据目录，HVUE Calici shortcut 检查也被公开 CSV 缺失 taxonomy metadata 所阻断。

| 评估目标 | Benchmark | 任务范围 | 期望结果 |
|:---|:---|:---|:---|
| Forget | HVUE | human host tropism; human-virus pathogenicity; Coronaviridae / Orthomyxoviridae / Caliciviridae transmissibility | 相对 base model 明显下降 |
| Retain: general genomics | GUE | promoter, splice site, TF binding, chromatin accessibility | 相对 base model 尽量保持 |
| Retain: viral proxy | vGUE | host-range / DNA-vs-RNA / HIV / SARS-CoV-2 tasks | 计划中；当前仓库未落地 |

**辅助诊断**：val/test split 上的 Phase 1 probe AUROC 和 forget/retain perplexity 仍保留，用于分析方法是否破坏 host-tropism 表示、是否造成 generation-objective collateral damage，但不能单独支撑 selective unlearning 结论。

**方法**：
- **GD（Gradient Difference）**：最大化 forget CE loss，最小化 retain CE loss
- **RMU（Representation Misdirection for Unlearning）**：将 forget 集在 target layer 的激活推向随机方向，同时保持 retain 集激活与参考模型一致

**Condition**：
- `full`：更新全部 32 层
- `localized`：仅更新 layers 3–9（Phase 1 activation patching 确定的因果层）
- `probe`：更新 layers 0–10（probe salience baseline，仅 GD）
- `random`：更新 layers 11–30 中随机选取的 7 层（负对照，与 localized 参数量相当）

**超参数**：steps=200, lr=1e-5, batch_size=2, max_length=512

---

## Benchmark 评估协议

新增 `phase2/eval_benchmarks.py`，用于对 base model 和每个 unlearned checkpoint 运行同一套外部 benchmark 评估。输入 manifest 需要包含：

```text
benchmark,task,split,sequence,label
```

可选列：`group`、`family`、`id`。当前仓库重建出的 manifest 显式提供 `group=hvue_forget|gue_retain`。也就是说，现阶段主 benchmark 是 `HVUE human-virus forget` 对 `GUE general-genomics retain`，而不是 `HVUE forget + HVUE retain + GUE retain` 三组都已落地。最新审计显示 `data/benchmarks/hvue_gue_manifest.csv` 与本地 HVUE 原始文件的 split 覆盖已经对齐。

每个 task 的流程：
1. 加载 base model 或应用 checkpoint delta。
2. 在指定 layers（默认 3–9）抽取 frozen Evo mean-pooled representations。
3. 对每个 task/layer 训练 L2 logistic regression probe，C-grid 为 `{0.001, 0.01, 0.1, 1.0}`，validation split 选 C。
4. 在 test split 上报告 accuracy、macro-F1、binary AUROC 或 multiclass macro-AUROC。
5. 汇总 forget group 和 retain group 的平均 benchmark score。

选择 unlearning 方法时应看相对 base model 的 Δ：
- HVUE human-virus forget score 下降越多越好；
- GUE retain score 越接近 base 越好；
- 若 PPL 保持但 GUE retain benchmark 下降，仍然说明 retain 失败；
- 若 PPL 变差但 GUE retain benchmark 保持，说明 PPL 主要是诊断信号而非最终 retain 指标。

### Step 1 可执行性结论

- `BVBRC Calici pathogenicity` 和 `Calici transmissibility` 两个公开 HVUE 任务当前都只有 `sequence,label`，没有 family/genus/species/accession 等 taxonomy 列。
- 两个任务本身又已经限定在 `Caliciviridae` 内，因此严格的 `family-held-out` split 在定义上不成立。
- 因而，当前公开 CSV 无法支持你要求的 taxonomy-shortcut 检查。若要继续，需要补充外部 taxonomy metadata，然后改成 `genus-held-out` 或 `species-held-out` 检查。
- 在该元数据补齐前，可靠 primary forget benchmark 应优先使用 `host tropism` 和 `CINI pathogenicity`。

### vGUE / Vir2vec 状态

- 已检查 `Vir2vec` 公共仓库，结论记录在 `data/benchmarks/vgue_from_vir2vec_audit.json`。
- 该仓库提供的是 train/validation/test accession split（BV-BRC / GISAID / HBVdb / LANL-HIV-DB / NCBI Virus）和 embedding/use-case 脚本。
- 它**没有**提供可直接并入本仓库 `eval_benchmarks.py` 的统一 `vGUE` task table（即 `benchmark,task,split,sequence,label` 形式的数据）。
- 因此，`Vir2vec` 可以作为 vGUE 的上游来源，但不能把“vGUE 已集成”这件事视为已完成。要真正接入，还需要任务级 sequence+label 表和 accession-to-task 的映射。

### 建议的替代实验顺序

1. 先把真正的 `family-held-out` 放到你自己的 `host_tropism` 主数据集或 `CINI` 这种混合家族任务上做。
2. 对单家族 Calici 任务，改做 `genus-held-out` 或 `species-held-out`。
3. Retain 侧先继续使用 `GUE`；把 `Vir2vec/vGUE` 标为下一阶段数据工程任务，而不是当前 sweep 的硬阻塞项。
4. 等 vGUE task table 补齐后，再把 retain 评估升级成 `vGUE + GUE`。


---

## 当前内部诊断结果（legacy，不作为最终主结论）

### Phase 1 probe AUROC（layers 3–9 均值）

| 方法 | Condition | Probe AUROC 均值 | Δ vs 基线 | forget_ppl diagnostic | retain_ppl diagnostic |
|:----:|:---------:|:----------------:|:---------:|:---------------------:|:---------------------:|
| 基线 | — | 0.844 | — | ~4.2 | ~4.2 |
| **GD** | **localized** | **0.555** | **-0.289** | 20.38 | 15.70 |
| GD | full | 0.524 | -0.321 | 31.21 | 37.90 |
| GD | random | 0.846 | +0.002 | 4.20 | 4.23 |
| **RMU** | **full** | **0.700** | **-0.144** | 4.45 | 4.48 |
| RMU | localized | 0.765 | -0.079 | 4.39 | 4.42 |
| RMU | random | 0.846 | +0.002 | 4.21 | 4.26 |

### 逐层 probe AUROC（test set）

| Layer | 基线 | GD full | GD local | GD rand | RMU full | RMU local | RMU rand |
|:-----:|:----:|:-------:|:--------:|:-------:|:--------:|:---------:|:--------:|
| 0 | 0.870 | 0.557 | 0.877 | 0.877 | 0.667 | 0.877 | 0.877 |
| 1 | 0.865 | 0.602 | 0.865 | 0.865 | 0.723 | 0.865 | 0.865 |
| 2 | 0.859 | 0.563 | 0.862 | 0.862 | 0.756 | 0.862 | 0.862 |
| 3 | 0.854 | 0.527 | 0.673 | 0.856 | 0.723 | 0.815 | 0.856 |
| 4 | 0.855 | 0.533 | 0.536 | 0.855 | 0.706 | 0.789 | 0.855 |
| 5 | 0.849 | 0.530 | 0.578 | 0.854 | 0.661 | 0.771 | 0.854 |
| 6 | 0.853 | 0.514 | 0.617 | 0.855 | 0.738 | 0.806 | 0.855 |
| 7 | 0.859 | 0.511 | 0.583 | 0.860 | 0.724 | 0.740 | 0.860 |
| 8 | 0.801 | 0.536 | 0.435 | 0.805 | 0.685 | 0.710 | 0.805 |
| 9 | 0.838 | 0.515 | 0.466 | 0.842 | 0.667 | 0.726 | 0.842 |
| 10 | 0.812 | 0.497 | 0.435 | 0.819 | 0.646 | 0.663 | 0.819 |

---

## 诊断性发现

### 1. Random condition 验证了 Phase 1 的因果分析

两种方法的 `random` condition（更新 layers 11–30）host-tropism probe AUROC 几乎不变（Δ ≈ +0.002），forget/retain perplexity 也与基线相同。这说明：
- 改动非因果层对当前 host-tropism diagnostic 基本无影响；
- Phase 1 activation patching 识别的 layers 3–9 是更合理的 intervention target；
- 遗忘效果依赖于是否触及因果层，而不只是参数量。

### 2. GD 遗忘信号强但 collateral damage 风险大

- `GD localized`：probe AUROC 降幅最大（-0.289），但 retain_ppl 从 4.2 升至 15.7。
- `GD full`：probe AUROC 降幅更大（-0.321），但 retain_ppl 升至 37.9。
- 解释：GD 通过最大化 forget CE loss 破坏 human-tropic 序列预测能力，容易直接干扰 generation objective。

这些只是内部诊断。最终是否 retain 失败，需要看 GUE benchmark 是否下降。

### 3. RMU 诊断上副作用小但遗忘偏弱

- `RMU full`：probe AUROC 降幅 -0.144，retain_ppl 仅升至 4.48。
- `RMU localized`：遗忘偏弱（-0.079），retain diagnostic 保持。
- 解释：RMU 在表示空间操作，不直接最大化 CE loss，因此 PPL 副作用更小。

最终主结论应由 HVUE forget drop 与 GUE retain preservation 的组合决定，而不是由 retain_ppl 单独决定。

### 4. Layers 0–2 的 probe AUROC 在 localized/random condition 下不变

`GD localized` 和 `RMU localized` 均未改动 layers 0–2，这些层的 AUROC 保持在 0.86–0.88。再次印证 Phase 1 的发现：早期层的表示虽然线性可分，但不是主要因果 intervention target。

---

## 对 Phase 3 的预测

在 HVUE + GUE 主评估完成前，Phase 3 的 attack robustness 只能作为内部 probe-based 初步分析。优先比较：

| 方法 | Condition | 预期 | 需要 benchmark 验证的点 |
|:----:|:---------:|:----:|:----------------------|
| GD | localized | forget diagnostic 强但 retain 风险高 | GUE 是否真的下降 |
| GD | full | diagnostic 上模型退化严重 | GUE 是否大面积受损 |
| RMU | full | forget/retain diagnostic 平衡 | HVUE human-virus forget 是否下降足够 |
| RMU | localized | retain diagnostic 好但 forget 弱 | 是否需要增加 steer_coef 或 steps |

---

## 后续改进方向

1. 先运行 `bash phase2/run.sh audit`，确认本地数据/manifest/ckpt 状态与 `data/phase2/experiment_audit.json` 一致。
2. 重建或修复 HVUE/GUE manifest 后，再运行 `bash phase2/run.sh benchmarks`，并优先报告相对 base 的 HVUE forget Δ 与 GUE retain Δ。
3. 在 vGUE 落地前，retain 侧仍以 GUE 为主；不要把 `vGUE + GUE` 写成已完成实验。
4. 根据结果再决定是否调参：RMU localized 可尝试增大 steer_coef 或 steps；GD 可尝试增大 alpha_retain，但只有 GUE retain benchmarks 保持时才算成功。
5. 若后续补充 taxonomy metadata，可把 Step 1 升级成 Calici `genus/species-held-out` shortcut audit；若补齐 vGUE，再把 retain 框架升级成 `vGUE + GUE`。
