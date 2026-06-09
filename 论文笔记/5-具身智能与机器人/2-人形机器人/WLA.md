---
title: "World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis"
method_name: "WLA"
authors: [Yi Yang, Zhihong Liu, Siqi Kou, Yiyang Chen, Yanzhe Hu, Jianbo Zhou, Boyuan Zhao, Zhijie Wei, Xiao Xia, Xueqi Li, Pengfei Liu, Zhijie Deng]
year: 2026
venue: arXiv
tags: [world-language-action, embodied-foundation-model, autoregressive-transformer, world-model, vla, test-time-scaling, cross-embodiment]
zotero_collection: 5-具身智能与机器人/2-人形机器人
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.05979v1
created: 2026-06-08
---

# 论文笔记：World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 上海交通大学 + 合作机构 |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[Pi05]] / π₀ / Motus / Fast-WAM / X-VLA |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05979) / [HTML](https://arxiv.org/html/2606.05979v1) |

---

## 一句话总结

> 把 [[World-Action Model|WAM]] 的"视觉未来预测"与 [[VLA]] 的"语言子任务推理"塞进同一根 AR Transformer，预测 *文本子任务 + 子目标图像 + 动作块* 三股流，无 embodied 预训练就在 RoboTwin 拿 92.94%、RMBench 翻倍至 56.5%。

---

## 核心贡献

1. **WLA 新范式**：首次把 *world modeling + language reasoning + action synthesis* 统一进单一 AR Transformer，把"下一状态"拆成**语义级文本子任务**和**物理级视觉 latent**两路互补预测。
2. **隐式世界建模**：World Expert 仅在训练时反传梯度共享给 backbone，**推理时可整段移除**，得到 ~40ms 的低延迟同时保留长程规划能力。
3. **Test-Time Scaling (TTS) 推理范式**：采样 $K$ 个动作候选 → World Expert 想象未来帧 → value model 打分 → 选最高分执行，把"想象式 MPC"塞进通用机器人策略。
4. **跨实体视频学习**：动作-free 同/跨实体视频可直接喂给 World Expert，把未见任务成功率从 13%/11.6% 拉到 34.4%/30.0%。
5. **强 OOD 实证**：RMBench 平均 **56.5%**（次优 28.5%），4 个真机长程任务在 OOD object/scene 下保持鲁棒，且推理速度比 Motus 快约 40×。

---

## 问题背景

### 要解决的问题

如何把 [[World Model|世界模型]] 的"物理预测能力"和 [[VLA]] 的"指令理解+长程分解"塞进同一个 [[Embodied AI|embodied]] foundation model，让单一模型既能从海量 egocentric 视频里学动力学，又能从语言里学高层意图。

### 现有方法的局限

- **WAM 类**（[[World-Action Model]] / [[Video World Model]]）：把建模负担压到像素级未来预测上，**缺语义抽象**，长程任务时模型必须复现所有低层视觉细节，语义推理能力被挤压。
- **VLA 类**（[[Pi05|π₀.₅]] / OpenVLA）：拥有语言推理但**缺物理动力学监督**，OOD 时表现脆弱；对动作-free 视频的利用接近 0。
- 异构数据（cross-embodiment 视频、无标注视频）几乎无法被现有 VLA/WAM 同时吸收。

### 本文的动机

把"下一状态 $s_{t+1}$"分解为 *semantic 子任务 $\mathcal{S}_t$* 与 *physical latent $\mathbf{h}_t$*，让 backbone 同时承担"想下一步该做什么"（[[VLM]] 风格语言推理）和"下一帧长啥样"（[[Video World Model]] 风格视觉想象），并通过 [[Causal Attention|因果注意力]] + meta-query 实现跨任务共享。

---

## 方法详解

### 模型架构

WLA 采用 **AR Transformer + 双 Expert** 架构（区别于 [[Mixture-of-Transformers]] 风格 WAM 的双向 [[Diffusion Model|diffusion]] backbone）：

- **输入**:
  - 自然语言指令 $\ell$
  - 历史/当前观测 $\mathbf{o}_{t-h}, \mathbf{o}_t$
  - 本体感觉状态 $\mathbf{q}_t$（proprioception）
  - 跨任务 [[Causal Attention|memory buffer]] $\mathcal{M}$
- **Backbone**: **RynnBrain-2B**（2.1B 参 [[VLM]]）初始化的 AR Transformer，纯自回归、纯 causal mask
- **核心机制**: **meta-queries** $\mathbf{Q}$ 附加在 context 末尾，通过 causal attention 聚合上下文得到 latent $\mathbf{h}_t$
- **World Expert** $f_{wm}$: 接 $\mathbf{h}_t$，预测目标帧 $\mathbf{o}_{t+n}$ 的 [[VAE]] 特征（不预测原像素），训练时反传，**推理时可关掉**
- **Action Expert** $f_{act}$: 390M / 28-layer **flow-matching head**，从 $\mathbf{h}_t$ + 状态 $\mathbf{q}_t$ 解码 [[Action Chunking|动作块]] $\mathbf{a}_{t:t+n}$
- **总参数**: ~2B active（被作者自标为 *WLA-0* baseline 版本）

### 核心模块

#### 模块 1: 文本意图学习 (Textual Intention Learning)

**设计动机**: 利用 [[VLM]] 的语言先验，把长程任务自然分解为可被高层规划复用的子任务序列。

**具体实现**:
- 自回归预测**连续子任务窗口** $\mathcal{S}_t$，跨度对齐动作 horizon $[t, t+n]$
- 子任务为 **free-form text**，不依赖固定词表，由训练标注从原指令分段得到
- 维护跨步 memory buffer $\mathcal{M}$ 保持长程任务一致性
- 用 [[VLM]] 风格的 next-token 监督（[[Teacher Forcing]]）

#### 模块 2: 物理动力学建模 (Physical Dynamics Modeling)

**设计动机**: 拒绝 WAM 式"全帧重建"，把物理状态压成轻量 latent，让 backbone 不被低层像素拖死。

**具体实现**:
- 在 context 尾部追加 $|\mathbf{Q}|$ 个 meta-query token，经 causal attention 后输出 $\mathbf{h}_t$
- $\mathbf{h}_t$ 不预测全 video clip，只预测**单目标帧 $\mathbf{o}_{t+n}$**（节约算力 + 抑制 [[Exposure Bias]]）
- World Expert 输出 [[VAE]] 特征空间表示，因为"语义抽象由 backbone 完成，无需再叠 [[VLM]] 归纳偏置"
- 训练完 World Expert 即可在 inference 移除，[[VAE]] 解码器也无需加载

#### 模块 3: Action Expert + Flow Matching

- 接 $\mathbf{h}_t$ 与 $\mathbf{q}_t$，用 [[Flow Matching]] 头部生成 chunk $\mathbf{a}_{t:t+n}$
- World modeling 通过共享 backbone 参数**间接**引导 action 预测，论文称之为 *implicit world modeling*

#### 模块 4: Test-Time Scaling (TTS) 推理

- 采样 $K$ 个动作候选 $\{\mathbf{a}^{(k)}_{t:t+n}\}$（不同随机种子）
- 对每个候选用 World Expert 想象未来帧 $\hat{\mathbf{o}}^{(k)}_{t+n}$
- 用预训练 **value model**（基于 WLA-0 rollout + 二元 success label 微调）打分
- 选 argmax，可自回归延长 imagination horizon

---

## 关键公式

### 公式 1: [[VLM|Subtask Prediction]]（文本子任务）

$$
\mathcal{S}_t = f(\mathbf{o}_{t-h},\, \mathbf{o}_t,\, \ell,\, \mathcal{M})
$$

**含义**: 给定历史/当前观测、指令和记忆，预测覆盖 horizon $[t, t+n]$ 的连续子任务窗口。

**符号说明**:
- $\mathcal{S}_t$: 连续子任务序列（free-form text token 序列）
- $\mathbf{o}_{t-h}, \mathbf{o}_t$: 历史与当前观测帧
- $\ell$: 原始自然语言指令
- $\mathcal{M}$: 跨任务 memory buffer

### 公式 2: [[Causal Attention|Physical Dynamics Latent]]

$$
\mathbf{h}_t = f(\mathbf{o}_{t-h},\, \mathbf{o}_t,\, \ell,\, \mathcal{M},\, \mathcal{S}_t,\, \mathbf{Q})
$$

**含义**: meta-queries $\mathbf{Q}$ 通过 causal attention 聚合所有上下文（含语义子任务），输出物理动力学 latent $\mathbf{h}_t$。

**符号说明**:
- $\mathbf{Q}$: 可学习的 meta-query token（条数控制 $\mathbf{h}_t$ 维度）
- $\mathbf{h}_t$: 紧凑物理 latent，下游同时驱动 World Expert 和 Action Expert

### 公式 3: [[Video World Model|World Expert]] 未来帧预测

$$
\mathbf{o}_{t+n} = f_{wm}(\mathbf{h}_t,\, \mathbf{o}_t)
$$

**含义**: World Expert 接物理 latent + 当前帧，预测 $n$ 步后目标帧的 [[VAE]] 特征。**仅训练用**，推理可移除。

**符号说明**:
- $f_{wm}$: World Expert（独立 Transformer 头）
- $\mathbf{o}_{t+n}$: 输出为 VAE feature，非原像素

### 公式 4: [[Action Chunking|Action Expert]] 动作块生成

$$
\mathbf{a}_{t:t+n} = f_{act}(\mathbf{h}_t,\, \mathbf{q}_t)
$$

**含义**: Action Expert 用 [[Flow Matching]] 头从物理 latent 和本体状态生成 $n$ 步动作 chunk。

**符号说明**:
- $f_{act}$: 390M / 28-layer flow-matching head
- $\mathbf{q}_t$: 本体感觉状态（关节位置/速度）
- $\mathbf{a}_{t:t+n}$: 连续 $n$ 步动作块

### 公式 5: 联合训练损失

$$
\mathcal{L} = \mathcal{L}_{\text{act}} + \alpha\, \mathcal{L}_{\text{wm}} + \beta\, \mathcal{L}_{\text{lang}}
$$

**含义**: 三项联合：动作回归 + 世界模型重建 + 语言子任务交叉熵。

**符号说明**:
- $\mathcal{L}_{\text{act}}$: flow-matching 动作回归损失
- $\mathcal{L}_{\text{wm}}$: World Expert VAE-feature MSE 重建损失
- $\mathcal{L}_{\text{lang}}$: 文本子任务 next-token CE 损失
- $\alpha = 0.1$, $\beta = 0.005$（论文实验值）

### 公式 6: TTS 推理选择规则

$$
k^{\star} = \arg\max_{k \in [K]}\; V_\phi\!\left(\hat{\mathbf{o}}^{(k)}_{t+n}\right),\quad \hat{\mathbf{o}}^{(k)}_{t+n} = f_{wm}(\mathbf{h}^{(k)}_t,\, \mathbf{o}_t)
$$

**含义**: 在 $K$ 个动作候选对应的想象未来帧上用 value model $V_\phi$ 打分，选最高分动作 chunk 执行。

**符号说明**:
- $K$: 采样候选数（论文默认 8）
- $V_\phi$: 预训练值函数（WLA-0 rollout 上微调，binary success 标签）

---

## 关键图表

### Figure 1: 架构对比 (VLA vs WAM vs WLA)

![Figure 1](https://arxiv.org/html/2606.05979v1/x1.png)

**说明**: (a) 经典 [[VLA]]：图像+指令→动作；(b) [[World-Action Model|WAM]]：图像→未来帧+动作；(c) WLA：AR Transformer 同时预测**文本意图** + **物理动力学** + **动作**；(d) WLA-0 用 2B 参 + **无 embodied 预训练**就已能在 RoboTwin/LIBERO 击穿 SOTA。

### Figure 2: WLA 系统概览

![Figure 2](https://arxiv.org/html/2606.05979v1/x2.png)

**说明**: 三组件分工：AR Transformer backbone 做 next-state 预测、World Expert 做未来帧成像、Action Expert 做动作块生成。**TTS 模式**额外引入 value model 打分→执行 argmax 动作 chunk。

### Figure 3: 真机长程任务结果 + 推理效率

![Figure 3](https://arxiv.org/html/2606.05979v1/x3.png)

**说明**: 4 个长程双手任务（Unscrew Cap / Pack Object / Stack Cup / Dispose Trash）在 **standard / OOD object / OOD scenario** 三档评测下的成功次数；右侧推理延迟柱状图：WLA-0 **~40ms vs Motus ~1600ms**，差距 40×。

### Figure 4: Beat Block Hammer 三档可视化

![Figure 4](https://arxiv.org/html/2606.05979v1/x4.png)

**说明**: Beat Block Hammer 任务在 *Seen-Action* / *+Video* / *+Same-Embodiment Video* 三档下的执行轨迹对比，证明同实体视频监督显著提升未见任务表现。

### Figure 5: TTS 推理示意

![Figure 5](https://arxiv.org/html/2606.05979v1/x5.png)

**说明**: TTS 流程：采样 K 个动作 → World Expert 想象未来 → value model 打分 → 选 argmax；可自回归延长 horizon。

### Figure 6-9: RMBench 任务示意

![Figure 6 - Battery Try](https://arxiv.org/html/2606.05979v1/figures/battery_try.png)
![[WLA_fig7.png|600]]
![Figure 8 - Cover Blocks](https://arxiv.org/html/2606.05979v1/figures/cover_blocks.png)
![Figure 9 - Press Button](https://arxiv.org/html/2606.05979v1/figures/press_button.png)

**说明**: RMBench 四个高难度记忆依赖任务的代表帧 + 子任务分解，要求模型在多步骤中维持"目标已完成"的语义记忆。

### Figure 10: 真机推理 World Expert 想象 vs 真实

![Figure 10](https://arxiv.org/html/2606.05979v1/x6.png)

**说明**: 真机推理时 World Expert 预测的未来帧 vs ground-truth 对比，证明想象画面足够接近真实，为 TTS 的 value-model 打分提供可靠输入。

### Table 1: RoboTwin 2.0 + LIBERO 主表

| Method | Active Params | Embodied Pretraining | RoboTwin Clean | RoboTwin Rand. | LIBERO Avg. |
|--------|---------------|----------------------|----------------|----------------|-------------|
| π₀ | 3B | ✓ | 65.92 | 58.40 | 94.2 |
| [[Pi05\|π₀.₅]] | 3B | ✓ | 82.74 | 76.76 | 96.9 |
| Motus | 8B | ✓ | 88.66 | 87.02 | 97.7 |
| Lingbot-VA | 5B | ✓ | 92.90 | 91.50 | 98.5 |
| Fast-WAM | 6B | ✗ | 91.88 | 91.78 | 97.6 |
| **WLA-0** | **2B** | ✗ | **92.94** | 90.02 | **98.6** |
| − $\mathcal{L}_{wm}$ | 2B | ✗ | 90.98 | 89.34 | 97.9 |
| + TTS | 2B | ✗ | — | — | **98.9** |

**说明**: WLA-0 用 **2B 参数 + 零 embodied 预训练**追平/超过 5B/8B 重量级 baseline。删 $\mathcal{L}_{wm}$ 后 RoboTwin Clean 掉 2 个点，证明世界模型损失有效。

### Table 2: RMBench（长程记忆依赖）

| Method | Battery Try | Blocks Ranking | Cover Blocks | Press Button | Average |
|--------|-------------|----------------|--------------|--------------|---------|
| [[Pi05\|π₀.₅]] | 16% | 6% | 0% | 0% | 5.5% |
| X-VLA | 26% | 1% | 2% | 0% | 7.3% |
| Mem-0 | 28% | 18% | 68% | 0% | 28.5% |
| Fast-WAM | 16% | 37% | 0% | 0% | 13.3% |
| **WLA-0** | **45%** | 23% | **84%** | **74%** | **56.5%** |
| − $\mathcal{L}_{lang}$ | 38% | 12% | 18% | 1% | 17.3% |

**说明**: WLA-0 在记忆依赖任务上**接近翻倍**次优。**移除语言损失后 Press Button 几乎归零**——说明语义子任务对维持"已完成步骤"的内隐记忆至关重要。

### Table 3: 跨实体视频学习（RoboTwin 5 个未见任务）

| Task | Seen-Action | +Video | +Same-Emb. | +Cross-Emb. |
|------|-------------|--------|------------|-------------|
| Beat Block Hammer | 1/0 | 1/0 | 12/6 | 5/3 |
| Move Playingcard | 2/0 | 0/3 | 42/28 | 1/0 |
| Pick Diverse Bottles | 7/10 | 10/8 | 32/29 | 39/35 |
| Place Object Basket | 3/5 | 0/1 | 30/34 | 45/47 |
| Stack Bowls Three | 52/43 | 48/51 | 56/53 | 54/52 |
| **Average** | **13.0/11.6** | 11.8/12.6 | **34.4/30.0** | 28.8/27.4 |

**说明**: Seen-Action 基线 13% → 同实体视频 **34.4%** → 跨实体仍有 28.8%。证实 World Expert 能从动作-free 视频中**学到可迁移的物理先验**。

### Table 4: Single-Frame vs Multi-Frame 预测（LIBERO）

| 设置 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| **Single-frame** | **98.8** | **100.0** | **97.4** | **96.6** | **98.2** |
| Multi-frame | 95.4 | 96.6 | 93.2 | 91.4 | 94.2 |

**说明**: 反直觉但坚定的发现——只预测**单一目标帧** > 预测整段未来 clip，过密的视觉监督会反噬动作学习。

### Table 5: 真机长程任务（双手机器人）

| Task | Standard | OOD Obj. | OOD Scenario |
|------|----------|----------|--------------|
| Unscrew Cap | 7 | 3 | 6 |
| Pack Object | 7 | 5 | 4 |
| Stack Cup | 10 | 9 | 7 |
| Dispose Trash | 6 | 4 | 2 |
| **Average** | **7.5** | **5.25** | **4.75** |

**说明**: 真机 4 任务（每任务 10 试次）OOD object 仅掉 2.25 分、OOD scenario 掉 2.75 分，**对应在 standard 7.5/10 的基础上仍有 ~50% 鲁棒性**，相对 WAM baseline 大幅领先。

### Table 6: 人类 egocentric 视频学习

| 设置 | 5 任务平均 |
|------|-----------|
| Seen-Action 基线 | 13.0 / 11.6 |
| + Unseen Human-Ego 视频 | 7.8 / 7.8 |

**说明**: 加入人类 egocentric 视频**反而下降**——作者归因为人/机器人 embodiment 分布差距过大，是当前方法的真实限制。

### Table 7: RoboTwin 2.0 全 50 任务（节选）

WLA-0 在全部 50 个 manipulation 任务上 clean 平均 **92.94%** / randomized **90.02%**，逐任务对比中绝大多数超过 [[Pi05|π₀.₅]] / Motus。完整逐任务表见原文附录 D。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **RoboTwin 2.0** | 50 任务 × N demo | 双手机器人仿真，clean + randomized 两档 | 主训练 + 主评测 |
| **LIBERO** | 4 suite (Spatial/Object/Goal/Long) | 经典 [[VLA]] benchmark | 评测泛化能力 |
| **RMBench** | 4 任务 (Battery / Blocks / Cover / Press) | 强记忆依赖长程任务 | 评测长程语义记忆 |
| 真机长程 | 4 双手任务 × 10 trial × 3 setting | Unscrew / Pack / Stack / Trash | OOD 鲁棒性评测 |
| 跨实体视频 | RoboTwin 同/跨实体 + 人类 ego | 动作-free | World Expert 监督源 |

### 实现细节

- **Backbone**: **RynnBrain-2B**（2.1B 参 [[VLM]]），AR Transformer
- **Action Expert**: 390M / 28-layer **flow-matching head**
- **VAE**: 预训练 image VAE，World Expert 预测其 latent space
- **Optimizer**: AdamW，base LR $5 \times 10^{-5}$，min LR $5 \times 10^{-6}$，1k warm-up
- **Batch size**: 256–448（任务依赖）
- **Training steps**: RoboTwin/LIBERO 100k；RMBench 30k
- **Loss weights**: $\alpha = 0.1$（world model），$\beta = 0.005$（language）
- **硬件**: DeepSpeed 分布式训练；推理 RTX 5090
- **推理延迟**: 高效模式 ~40 ms（含 CUDA Graph + operator fusion），TTS 模式按 K 倍延长

### 可视化结果

- World Expert 想象帧与 ground-truth 接近，足以支撑 TTS 中 value 模型打分
- Dispose Trash 任务执行时间相对 WAM baseline 直接**减半**
- RMBench Press Button 子任务依赖"已按下按钮"的语义记忆，去掉 $\mathcal{L}_{lang}$ 后近乎归零

---

## 批判性思考

### 优点

1. **范式整合干净**：不是简单堆 head，而是真正用 AR Transformer + meta-query 把"语义子任务 / 物理 latent / 动作"三流统一进 next-state 预测语义。
2. **隐式世界模型**：训练用、推理可拆，绕过了 WAM 类 ~1.6s 推理延迟的工程毒瘤。
3. **TTS 巧用 World Expert**：把"训练阶段唯一资产"在 inference 复活当 MPC rollout 引擎，零额外仿真器。
4. **跨实体视频实证**：Table 3 用同/跨实体视频拉起 ~3× 未见任务成功率，是 WLA 范式最关键的实证支撑。
5. **2B 击穿 8B**：参数效率给 mobile/边缘部署留出空间。

### 局限性

1. **真机评测覆盖窄**：仅 1 个双手平台、4 个任务，足式/灵巧手/多机协同未触及。
2. **人类 ego 视频迁移失败**（Table 6）：声称的"从海量人类视频学"目前还是愿景而非现实。
3. **value model 依赖任务内 rollout**：TTS 的 value model 是用 WLA-0 自己的 rollout + 二元成功标签微调，**跨任务复用未验证**。
4. **隐式世界建模的语义可解释性弱**：[[VAE]] feature space 预测没有给出可视化诊断工具，难判断 backbone 真的学到了"动力学"还是"快捷特征"。
5. **+TTS 数据缺失**：Table 1 的 RoboTwin 列 TTS 一栏是 "—"，作者只给了 LIBERO 提升 0.3 个点。

### 潜在改进方向

1. value model 跨任务迁移：考察是否能训一个 task-agnostic value head 给 TTS 用。
2. 替换 [[VAE]] latent 为 [[FSQ]] / Cosmos Tokenizer 类离散 token，让 World Expert 走 next-token 预测，简化训练。
3. 用真实人类视频 + retargeting 桥接 embodiment gap，把 Table 6 救回来。
4. 与 [[OmniDreams]] / Cosmos3 类 omnimodal world model 对接：让 backbone 自带视频生成能力，World Expert 退化为投影头。

### 可复现性评估

- [ ] 代码开源（论文未提）
- [ ] 预训练模型（未提）
- [x] 训练细节完整（loss 系数、LR、steps 都列）
- [x] 数据集可获取（RoboTwin / LIBERO 公开；RMBench 需查）

---

## 关联笔记

### 基于

- [[VLM]]: backbone 用 2B VLM (RynnBrain) 初始化
- [[World-Action Model]]: 直接对标对象，WLA 把语义流加回来
- [[Video World Model]]: World Expert 是其退化版（单帧、latent-space）
- [[Flow Matching]]: Action Expert 的生成范式

### 对比

- [[Pi05|π₀.₅]]: 经典 VLA，WLA 在 RoboTwin / LIBERO 上全面超越
- [[World-Action Model|Fast-WAM]]: 同样无预训练的 WAM 基线，WLA 用更少参数追平
- [[OmniDreams]]: WAM 范式的另一支，参数 backbone 思路更激进
- Motus: pretrained 重量级 VLA，WLA 用 1/4 参数 + ~40× 推理速度抗衡

### 方法相关

- [[Causal Attention]]: meta-query 聚合上下文的核心
- [[Action Chunking]]: Action Expert 输出 chunk 而非单步
- [[Teacher Forcing]]: 文本子任务训练用 next-token 监督
- [[VAE]]: World Expert 预测的特征空间载体
- [[Exposure Bias]]: 单帧预测 vs 多帧预测的根本权衡来源

### 硬件/数据相关

- [[RoboTwin]]: 主仿真+评测平台
- [[LIBERO]]: 通用 VLA benchmark
- [[RMBench]]: 长程记忆 benchmark
- [[Seen-Action Baseline]]: 跨实体视频实验中的对照设置

---

## 速查卡片

> [!summary] WLA: World-Language-Action
> - **核心**: AR Transformer + meta-query 同时预测 *文本子任务 + 视觉 latent + 动作块*
> - **方法**: 2B RynnBrain backbone + 390M flow-matching Action Expert + 可移除的 World Expert
> - **结果**: RoboTwin 92.94 / LIBERO 98.6 / RMBench 56.5%，~40ms 推理（比 Motus 快 40×）
> - **杀手特性**: 隐式世界建模（推理可拆）+ TTS 想象式 MPC + 跨实体视频学习
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-08*
