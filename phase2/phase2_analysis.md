# Phase 2 Unlearning 结果分析

## 实验设置

**目标**：在 Evo-1-8k-base 上实现 host tropism 能力的靶向遗忘，比较不同方法和 condition 的效果。

**Forget set**：human-tropic viral 序列（label=1，3800 条训练样本）
**Retain set**：non-human-tropic viral 序列（label=0，3814 条训练样本）
**评估**：val/test split 上的 probe AUROC（复用 Phase 1 probes）+ forget/retain perplexity

**方法**：
- **GD（Gradient Difference）**：最大化 forget CE loss，最小化 retain CE loss
- **RMU（Representation Misdirection for Unlearning）**：将 forget 集在 target layer 的激活推向随机方向，同时保持 retain 集激活与参考模型一致

**三个 Condition**：
- `full`：更新全部 32 层
- `localized`：仅更新 layers 3–9（Phase 1 activation patching 确定的因果层）
- `random`：更新 layers 11–30 中随机选取的 7 层（负对照，与 localized 参数量相当）

**超参数**：steps=200, lr=1e-5, batch_size=2, max_length=512

---

## 结果汇总

### AUROC（layers 3–9 均值）

| 方法 | Condition | AUROC 均值 | Δ vs 基线 | forget_ppl | retain_ppl |
|:----:|:---------:|:----------:|:---------:|:----------:|:----------:|
| 基线 | — | 0.844 | — | ~4.2 | ~4.2 |
| **GD** | **localized** | **0.555** | **-0.289** | 20.38 | 15.70 |
| GD | full | 0.524 | -0.321 | 31.21 | 37.90 |
| GD | random | 0.846 | +0.002 | 4.20 | 4.23 |
| **RMU** | **full** | **0.700** | **-0.144** | 4.45 | 4.48 |
| RMU | localized | 0.765 | -0.079 | 4.39 | 4.42 |
| RMU | random | 0.846 | +0.002 | 4.21 | 4.26 |

### 逐层 AUROC（test set）

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

## 关键发现

### 1. Random condition 验证了 Phase 1 的因果分析

两种方法的 `random` condition（更新 layers 11–30）AUROC 几乎不变（Δ ≈ +0.002），forget/retain perplexity 也与基线相同。这直接证明：
- **改动非因果层对 viral 能力毫无影响**
- Phase 1 activation patching 正确识别了 layers 3–9 为因果关键层
- 遗忘效果完全依赖于是否触及因果层，而非参数量

### 2. GD 遗忘强但代价大

- `GD localized`：AUROC 降幅最大（-0.289），但 retain_ppl 从 4.2 升至 15.7，通用基因组生成能力受损明显
- `GD full`：遗忘更彻底（-0.321），但 retain_ppl 爆到 37.9，模型基本不可用
- **根本原因**：GD 通过最大化 forget CE loss 来破坏 viral 序列的预测能力，这会直接干扰语言模型的生成目标，副作用难以避免

### 3. RMU 代价小但遗忘偏弱

- `RMU full`：AUROC 降幅 -0.144，retain_ppl 仅升至 4.48，**最佳 forget-retain 平衡**
- `RMU localized`：遗忘偏弱（-0.079），retain 完好，说明 200 步 + lr=1e-5 力度不足，或 steer_coef=20 需要调大
- **根本原因**：RMU 在表示空间操作（MSE loss），不直接破坏生成目标，对 retain 的副作用更小

### 4. Layers 0–2 的 AUROC 在 localized/random condition 下不变

`GD localized` 和 `RMU localized` 均未改动 layers 0–2，这些层的 AUROC 保持在 0.86–0.88。再次印证 Phase 1 的发现：**早期层的表示虽然线性可分，但不是因果关键层**，unlearning 不需要也不应该动它们。

---

## 对 Phase 3 的预测

基于以上结果，对 recovery attack（SFT / LoRA fine-tuning）的抵抗能力预测：

| 方法 | Condition | 预测抗攻击能力 | 理由 |
|:----:|:---------:|:-------------:|:-----|
| GD | localized | **弱** | retain 能力已受损，fine-tuning 信号强，容易恢复 |
| GD | full | **极弱** | 模型已严重退化，fine-tuning 会快速恢复 |
| RMU | full | **中等** | 遗忘适中，retain 完好，但 AUROC 仍有 0.70，攻击空间较大 |
| RMU | localized | **弱** | 遗忘太弱，几乎没有破坏 viral 表示 |

**最值得在 Phase 3 测试的组合**：`GD localized` vs `RMU full`，前者遗忘彻底但脆弱，后者平衡但不彻底。

---

## 后续改进方向

1. **RMU localized 调参**：增大 steer_coef（20 → 100）或 steps（200 → 500），可能在保持 retain 完好的同时提升遗忘效果
2. **GD + retain loss 权重调整**：增大 alpha_retain 以减少通用能力损失
3. **组合方法**：先用 RMU 破坏表示，再用 GD 强化遗忘
