# Activation Patching 结果分析

## 实验设计

**目标**：用 activation patching 找出对 host tropism（human vs non-human viral）能力**因果关键**的层，为 Phase 2 unlearning 选定目标层。

**方法**：对 source/target 配对样本，分别替换每一层的隐藏激活，测量两个指标变化：
- `delta_human_prob`：probe 预测概率的变化（衡量表示空间的因果效应）
- `delta_perplexity`：next-token 预测损失的变化（衡量对最终输出的影响）

**两个方向**：
- `nonhuman_to_human`：将 nonhuman 序列的激活替换进 human 序列的前向过程，看 human 预测概率是否下降
- `human_to_nonhuman`：反向操作

**配置**：n_pairs=16, max_length=512, split=test, 模型 Evo-1-8k-base (32 层 StripedHyena)

---

## 主要发现

### 1. Layer 6 是最强因果层

| Layer | nh→h \|Δprob\| | h→nh \|Δprob\| | Mean | Test AUROC (probe) |
|:-----:|:---------------:|:---------------:|:------:|:------------------:|
| **6** | **0.441** | **0.269** | **0.355** | 0.853 |
| **8** | 0.286 | 0.266 | 0.276 | 0.801 |
| **4** | 0.176 | 0.154 | 0.165 | 0.855 |
| **9** | 0.170 | 0.130 | 0.150 | 0.838 |
| **3** | 0.116 | 0.139 | 0.127 | 0.854 |
| 5 | 0.155 | 0.072 | 0.114 | 0.849 |
| 10 | 0.000 | 0.063 | 0.031 | 0.812 |
| 2 | 0.018 | 0.016 | 0.017 | 0.859 |
| 1 | 0.007 | 0.015 | 0.011 | 0.865 |
| 0 | 0.0001 | 0.0003 | 0.0002 | 0.870 |

- **layer 6**：双向均显著（nh→h: -0.441，h→nh: +0.269），说明这一层的激活承载了最多的 host tropism 信息
- **layer 8**：双向效应均衡（约 0.27），与 layer 6 同为关键层
- **layers 3, 4, 5, 9**：中等效应（0.11–0.17）

### 2. Probe AUROC ≠ 因果重要性

Layers 0–2 的 probe AUROC 很高（0.86–0.87），但 **patching 效应几乎为零**（\|Δprob\| < 0.02）。

**解读**：这些早期层的表示虽然**线性可分**，但替换它们并不能改变下游决策。可能原因：
- 早期层包含表面统计特征（GC、k-mer），probe 能利用它们做分类，但模型本身不依赖它们做生成
- 后续层会从原始 token 重新构造这些特征，patching 被"覆盖"

**含义**：probe 找出的"可分层"和 patching 找出的"因果层"不重合。**Phase 2 unlearning 应优先针对因果层（3–9），而不是 probe 最强的层（0）**。

### 3. Perplexity Delta 跨层近乎恒定

| 方向 | Δppl 均值 | Δppl 标准差 |
|:---:|:---:|:---:|
| nonhuman → human | 0.211 | 0.0001 |
| human → nonhuman | 1.809 | 0.0000 |

所有层的 patched perplexity **完全相同**（小数点后 4 位）。这表明：
- **单层 patching 不影响最终输出 loss**——模型后续层会"修复"被替换的激活
- delta_loss 反映的是 source/target 序列本身的差异（human 序列总体困惑度低，nonhuman 较高），而不是 patching 的效应

**对 unlearning 的启示**：
- 如果只改动单层权重，模型可以通过其他层补偿（绕过 unlearning）
- **Phase 2 unlearning 必须同时覆盖多层（layers 3–9）**，而不是只针对单层

### 4. Layer 11+ 数值不稳定

- 从 layer 10（L2 ≈ 257）跳到 layer 11（L2 ≈ 1.8M），再到 layer 13（L2 ≈ 7.3×10⁷）
- 这些层 probe AUROC 也跌到 0.60–0.69
- Patching 效应全部归零（probe saturate 到 0 或 1）

**含义**：后期层在 bfloat16 下的激活幅值发散（可能是 StripedHyena 长卷积的累积效应），probe 在这些层不可信。**Phase 2 评估和操作都应限于 layers 0–10**。

---

## 对 Phase 2 的具体指导

### Unlearning 目标层优先级
1. **Layer 6**（首选，因果效应最强）
2. **Layer 8**（次选，双向效应均衡）
3. **Layers 3–5, 9**（次次选，中等效应）

### 三种条件设计
1. **Full-model**：对照组，更新所有 32 层
2. **Localized layers**：限定更新 **layers 3–9**（基于本分析，比原 0–10 更精准）
3. **Random layers**：从 layers 11–31 随机选 7 层（与 localized 数量匹配，作为负对照）

### 评估策略
- 复用 Phase 1 的 probes，在 unlearned 模型上重新提取激活计算 AUROC
- **重点看 layers 3–9 的 AUROC 下降程度**，layers 0–2 的 AUROC 应保持（早期表层特征不应破坏，否则会损害通用基因组能力）
- Perplexity 评估也要做（retain set），确保 unlearning 不破坏一般生成能力

### 避免的陷阱
- 不要只针对单层做 unlearning（被绕过的风险高）
- 不要相信 layers 11+ 的 probe 输出（数值不稳定）
- Probe AUROC 高不等于因果重要——optimization 信号应优先来自 layers 3–9
