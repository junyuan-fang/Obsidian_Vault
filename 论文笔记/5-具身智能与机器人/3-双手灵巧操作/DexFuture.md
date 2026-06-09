---
title: "DexFuture: Hierarchical Future-State Visuomotor Targeting for Bimanual Dexterous Tool Use"
method_name: "DexFuture"
authors: [Runfa Blark Li, Kuang-Ting Tu, Nikola Raicevic, Dwait Bhatt, Xinshuang Liu, Keito Suzuki, Ki Myung Brian Lee, Nikolay Atanasov, Truong Nguyen]
year: 2026
venue: arXiv
tags: [bimanual-dexterous-manipulation, world-model, visuomotor-policy, hierarchical-policy, tool-use, oakink2]
zotero_collection: 5-具身智能与机器人/3-双手灵巧操作
image_source: local
arxiv_html: https://arxiv.org/html/2606.05699v1
created: 2026-06-08
---

# 论文笔记：DexFuture

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UC San Diego（多位作者隶属 ECE/CSE） |
| 日期 | June 2026（arXiv 2026-06-04 submission） |
| 项目主页 | 暂未公开 |
| 对比基线 | [[PhysGraph]]（GT target oracle）/ [[ManipTrans]] / [[DexWM]] (CEM planning) |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05699) / [HTML](https://arxiv.org/html/2606.05699v1) |

---

## 一句话总结

> 用 hierarchical 视觉运动 [[Future-State Visuomotor Target Predictor|未来状态预测器]] 替换 demo 给的特权 target 与昂贵 [[Cross-Entropy Method (CEM)|CEM]] 规划，60 Hz 在 [[OakInk2]] 上拿到 90% 特权 oracle 性能。

---

## 核心贡献

1. **去特权目标**：把双手灵巧策略对 demonstration future state 的依赖,替换为可学习的高层 [[Future-State Visuomotor Target Predictor|视觉运动目标预测器]] $F_\theta$，仅靠 egocentric RGB + 本体/几何历史就能生成未来 hand-tool-object target 轨迹。
2. **分层解耦**：把"慢的长时程语义预测"与"快的接触控制"显式分开 —— 高层 sparse horizon $\mathcal{H}=\{0,2,4,\ldots,16\}$ 出粗 target，低层 [[Target-Conditioned Structured Dexterous Policy|目标条件结构化策略]] $\pi_\phi$ 用 [[Receding-Horizon Execution|recede-horizon]] 跟踪插值 target 做 60 Hz 接触控制。
3. **结构化 visuomotor token**：把 hand-link、tool、object 拆成"per-link 视觉交叉注意 + anchor-aligned 场景池化"的结构化嵌入，配合 [[Horizon-Conditioned Transformer|horizon-conditioned transformer]] + [[AdaLN|AdaLN]] 条件化，使同一个模型为不同未来步长 specialize 出不同 prediction。
4. **替代 [[Cross-Entropy Method (CEM)|CEM]] 规划**：与 DexWM-style CEM 比较，DexFuture 在去掉 ground-truth target 的情况下达到 83.49% SR、60 Hz，CEM 即使 fine-tune 也只有 69.57% SR、0.24 Hz（~250× 慢）。

---

## 问题背景

### 要解决的问题
双手灵巧工具使用（bimanual dexterous tool use）需要在高维 hand 空间和 hand-tool-object 接触动力学下产生稳定动作。学术界两条主流路线都有缺陷：
- **demo-driven 策略**（PhysGraph / ManipTrans）依赖未来 demonstration target $g^{demo}_{t+h}$ 作为输入，等价于"特权 oracle"，在真实部署时拿不到。
- **action-conditioned world model + 规划**（DexWM + [[Cross-Entropy Method (CEM)|CEM]]）要在线 rollout 高维动作序列，单步规划只有 0.24 Hz，远低于接触控制需要的 60 Hz。

### 现有方法的局限
- PhysGraph (No target) 性能塌方：average SR 仅 ~7%（在 cut/wipe/shear/pour 等任务上）。
- DexWM CEM 在 horizon=1 + finetune 才有 69.57% SR，把 horizon 推到 16 直接掉到 0%（长 horizon 下高维动作空间不可解），且速度极慢。
- 直接预测低层动作（[[Action Chunking|action chunking]]）面对长时程 contact-rich 任务时缺乏结构化中间表示，难以推广。

### 本文的动机
"动作"很难直接长 horizon 预测，但"未来 state 中关键 link 的位置/姿态/接触"在视觉里有充分线索；把这些"中间目标"显式预测出来，比预测原始动作序列容易得多，又能复用现有 PhysGraph 风格的 [[Target-Conditioned Structured Dexterous Policy|目标条件低层策略]]。

---

## 方法详解

### 模型架构

DexFuture 是 **hierarchical visuomotor system**：

- **输入**:
  - egocentric RGB $I_\tau$
  - 本体感知 + 几何历史 $p_\tau$（hand joint、tool/object pose anchor）
  - 历史窗口长度 $K$
- **高层 Future-State Visuomotor Target Predictor $F_\theta$**:
  - **Tokenizer**: 把每个 hand link 通过 local visual cross-attention + [[FiLM Modulation|FiLM]] 调制构造 per-link token；tool/object 用 anchor-aligned scene pooling
  - **Backbone**: [[Horizon-Conditioned Transformer|horizon-conditioned transformer]] $T_\theta$,以 $h \in \mathcal{H}=\{0,2,4,\ldots,16\}$ 为 conditioning 通过 [[AdaLN|AdaLN]] 注入
  - **输出**: 稀疏 horizon target $\{\hat{g}_{t+h}\}_{h\in \mathcal{H}}$
- **低层 Target-Conditioned Structured Dexterous Policy $\pi_\phi$**:
  - **Tokenizer**: 把当前 state $s_t$ 也拆 per-link + scene token
  - **条件**: 线性插值后的 target $\tilde{g}_{t+\delta} = \mathrm{Interp}(\{\hat{g}_{t+h}\}, \delta)$
  - **输出**: 高斯动作分布 $\mathcal{N}(\mu_\phi, \Sigma_\phi)$,通过 PPO 训练
- **执行**: [[Receding-Horizon Execution|receding horizon]] —— 慢频率刷新 target 轨迹,快频率（60 Hz）跟踪插值 target

### 核心模块

#### 模块1：结构化 Visuomotor Tokenization（Fig.2-A）

**设计动机**: hand-tool-object 三者关系强 spatial-temporal 耦合，naive flatten 视觉特征丢失结构。

**具体实现**:
- Raw visual feature $V_{raw} = \Psi(I_t)$（vision backbone）
- 对每个 hand link $i$，用其 3D 几何先验在 $V_{raw}$ 上做 local cross-attention,得到 $z^{hnd}_i$（用 [[FiLM Modulation|FiLM]] 注入 link 类型/手别 ID）
- tool / object 各自取 anchor 点（grasp point、contact anchor）聚合得到 $z^{tool}, z^{obj}$
- 拼接得到 $Z_t = \{z^{hnd}_i\}_{i=1}^{N_h} \cup \{z^{tool}, z^{obj}\}$

#### 模块2：Horizon-Conditioned Target Transformer（Fig.2-B）

**设计动机**: 单一模型要同时输出不同未来步长 $h$ 的 target，又不想为每个 $h$ 训一份。

**具体实现**:
- Memory projection: $X_\tau = W_{in} Z_\tau$,堆叠成 transformer memory $M$
- Future query: 用 horizon-specific learnable slot + frame embedding 初始化 $Y_h^0$
- 每个 block：self-attention → cross-attention(memory $M$) → FFN,前置 [[AdaLN|AdaLN]]
- 门控系数 $\alpha_{sa}, \alpha_{ca}, \alpha_{ff}$ 也由 $c_h$ 条件化,允许网络对"近未来"和"远未来" specialize 不同的更新强度
- 输出 $\hat{Z}_{t+h} = W_{out} Y_h^L$,再解码 $\hat{g}_{t+h} = D_\theta(\hat{Z}_{t+h})$

#### 模块3：Target-Conditioned Structured Dexterous Policy（Fig.2-C）

**设计动机**: 复用 [[PhysGraph]] 的 per-link transformer 结构,但去掉对 GT target 的依赖。

**具体实现**:
- 同样 per-link + scene 结构化 token,与插值后的 target $\tilde{g}_{t+\delta}$ 拼接
- transformer encoder 编码后送 policy head 输出 $\mathcal{N}(\mu_\phi, \Sigma_\phi)$
- 训练用 PPO + tracking reward（公式 43）：$r_t = \sum_m \beta_m \exp(-\alpha_m d_m(s_t, s^{demo}_t)) - \beta_e \|a_t\|^2$

#### 模块4：Receding-Horizon Execution（Fig.2-D）

- 每隔 $J$ 步触发一次高层预测 $\hat{g}_{t_j:t_j+H}$
- 中间步以 60 Hz 用线性插值后的 $\tilde{g}_{t_j+\delta}$ 喂给低层
- "慢预测 + 快跟踪"的频率解耦避免高层每步重算的开销

---

## 关键公式

### 公式1：[[Target-Conditioned Structured Dexterous Policy|目标条件策略]]（理想情形）

$$
a_t \sim \pi_\phi(\cdot | s_t,\ g^{demo}_{t+h})
$$

**含义**: PhysGraph 类策略的输入约定——需要未来 $h$ 步的 demonstration target 作为条件。

**符号说明**:
- $a_t$: 双手动作（hand joints + wrist 6DoF）
- $s_t$: robot-object state
- $g^{demo}_{t+h}$: 来自 demonstration 的未来特权 target（DexFuture 要替换的就是这一项）

### 公式2：[[Visuomotor History|视觉运动历史]]

$$
\mathcal{O}_{t-K:t} = \{ I_\tau,\ p_\tau \}_{\tau=t-K}^{t}
$$

**含义**: 高层预测器的输入只用 RGB + 本体/几何历史,不依赖 demo。

**符号说明**:
- $I_\tau$: egocentric RGB
- $p_\tau$: proprioceptive + geometric cue
- $K$: 历史长度

### 公式3：[[Future-State Visuomotor Target Predictor|未来视觉运动 target 预测]]

$$
\hat{g}_{t+h} = F_\theta(\mathcal{O}_{t-K:t};\ \mathcal{H})
$$

**含义**: 一次性输出 sparse horizon set $\mathcal{H} = \{0,2,4,\ldots,16\}$ 上的 target,避免每个 $h$ 单独前向。

### 公式4：[[Receding-Horizon Execution|插值执行]]

$$
a_{t+\delta} \sim \pi_\phi(\cdot | s_{t+\delta},\ \tilde{g}_{t+\delta}),\quad
\tilde{g}_{t+\delta} = \mathrm{Interp}(\{\hat{g}_{t+h}\}_{h\in\mathcal{H}},\ \delta)
$$

**含义**: sparse target 之间用线性插值,得到 60 Hz 控制频率的密集 target。

### 公式5：结构化嵌入

$$
Z_t = \{ z^{hnd}_i \}_{i=1}^{N_h} \cup \{ z^{tool}_t,\ z^{obj}_t \}
$$

**含义**: 每帧 token 集合 = hand link tokens + tool token + object token,显式编码三者结构。

### 公式6：[[Horizon-Conditioned Transformer|Horizon 条件 transformer]]

$$
\hat{Z}_{t+h} = T_\theta(Z_t,\ Z_{t-K:t},\ h),\quad h \in \mathcal{H}
$$

**含义**: 同一 transformer $T_\theta$ 凭借 horizon 条件 $h$ 输出不同未来 step 的嵌入。

### 公式7：Target Decoder

$$
\hat{g}_{t+h} = D_\theta(\hat{Z}_{t+h})
$$

**含义**: 把 transformer 输出嵌入解码到与 policy 接口兼容的 target 表示（pose / joint / contact）。

### 公式8：预测器损失（精简）

$$
\mathcal{L}_{pred} = \lambda_{state}\mathcal{L}_{state} + \lambda_{target}\mathcal{L}_{target}
$$

**含义**: 同时监督结构化 state 重建与 target 预测。

### 公式9：策略动作分布

$$
a_t \sim \mathcal{N}(\mu_\phi(s_t,\ \tilde{g}_t),\ \Sigma_\phi)
$$

**符号说明**: $\mu_\phi$ 由 transformer encoder 输出,$\Sigma_\phi$ 为可学习对角协方差。

### 公式10：完整 receding-horizon 执行

$$
\begin{aligned}
\hat{g}_{t_j:t_j+H} &= F_\theta(\mathcal{O}_{t_j-K:t_j};\ \mathcal{H}) \\
a_{t_j+\delta} &\sim \pi_\phi(\cdot|s_{t_j+\delta},\ \mathrm{Interp}(\hat{g}_{t_j:t_j+H},\ \delta))
\end{aligned}
$$

**含义**: 高层在 $t_j$ 触发,中间步插值给低层。

### 公式11：[[AdaLN|AdaLN]] 条件化

$$
\mathrm{AdaLN}(Y_h,\ c_h) = \mathrm{LN}(Y_h) \odot (1 + s(c_h)) + b(c_h)
$$

**含义**: 把 horizon 嵌入 $c_h$ 注入 LayerNorm 的 scale/shift,使 transformer 块输出依赖 $h$。

**符号说明**:
- $s(c_h)$, $b(c_h)$: 由 horizon 编码 $c_h$ 经 MLP 得到的 scale/shift
- $\odot$: 逐通道乘

### 公式12：完整预测器损失（含 latent consistency）

$$
\mathcal{L}_{pred} = \lambda_z\mathcal{L}_z + \lambda_{state}\mathcal{L}_{state} + \lambda_{target}\mathcal{L}_{target}
$$

**符号说明**:
- $\mathcal{L}_z$: latent consistency loss（预测嵌入 vs 真实嵌入）
- $\mathcal{L}_{state}$: 结构化 state 监督
- $\mathcal{L}_{target}$: target component（pose / 关节 / 接触点）回归损失

### 公式13：PPO Tracking Reward

$$
r_t = \sum_{m \in \mathcal{M}} \beta_m \exp\!\left(-\alpha_m\, d_m(s_t,\ s^{demo}_t)\right) - \beta_e \|a_t\|^2
$$

**含义**: 多 modality 跟踪奖励 + 动作平滑惩罚。

**符号说明**:
- $\mathcal{M}$: 监督模态集合（wrist pose / fingertip / object pose 等）
- $d_m$: 模态距离
- $\beta_m, \alpha_m$: 各模态权重 / 温度
- $\beta_e$: 动作 L2 惩罚系数

---

## 关键图表

### Figure 1: 系统概览

![[assets/DexFuture/x1.png]]

**说明**: DexFuture 把高层 [[Future-State Visuomotor Target Predictor|future-state visuomotor target predictor]] 与低层 [[Target-Conditioned Structured Dexterous Policy|target-conditioned structured dexterous policy]] 串联,使策略不再依赖 demonstration 的特权 target,而是从 visuomotor 历史预测出粗未来 target trajectory,供低层接触控制。

### Figure 2: 方法细节（A-D 四模块）

![[assets/DexFuture/x2.png]]

**说明**: A 模块展示结构化 visuomotor tokenization（per-link cross-attn + anchor-aligned pooling）;B 模块是 [[Horizon-Conditioned Transformer|horizon-conditioned target transformer]] 含 [[AdaLN|AdaLN]] 调制与 refinement;C 模块是 PhysGraph 风格 per-link policy 与 policy head;D 模块展示 [[Receding-Horizon Execution|receding-horizon 执行]] 与线性插值的 target 提供。

### Figure 3: 定性 rollout 对比

![[assets/DexFuture/x3.png]]

**说明**: 四行任务（chop knife/bread、fruit knife/apple、scissors/paper、small brush/whiteboard）× 四列方法（ManipTrans-GT / [[PhysGraph]]-GT / PhysGraph-NoTarget / **DexFuture-Pred**）。无 target 的 PhysGraph 频繁掉刀失稳;DexFuture 在没有特权信息的情况下与 oracle 几乎相当。

### Figure 4: DexWM-CEM 定性结果

![[assets/DexFuture/x4.png]]

**说明**: 四种 [[DexWM]] + [[Cross-Entropy Method (CEM)|CEM]] 配置（Finetune+H1 / NoFinetune+H1 / Finetune+H16 / NoFinetune+H16）的定性 rollout。只有 Finetune+H1 勉强可用,其它直接失败,说明长 horizon CEM 在高维动作空间不可解。

---

### Table 1: Policy Evaluation Results（含 7 个任务,完整原始表）

| Task ID | 任务/工具/对象 | Method | SR (%) | $E_t$ (cm) | $E_j$ (cm) | $E_{xt}$ (cm) |
|---------|---------------|--------|-------:|-----------:|-----------:|--------------:|
| 083f7@0 | cut / chop knife / bread | ManipTrans (GT) | 55.69 | 1.31 | 2.42 | 2.01 |
|         |                          | [[PhysGraph]] (GT) | 90.05 | 0.69 | 2.17 | 1.42 |
|         |                          | PhysGraph (No target) | 4.16 | 1.87 | 3.11 | 2.68 |
|         |                          | **DexFuture (Pred)** | **83.49** | **1.06** | **2.08** | **2.04** |
| 9fc3e@0 | cut / fruit knife / apple | ManipTrans (GT) | 70.58 | 0.84 | 2.99 | 1.97 |
|         |                          | PhysGraph (GT) | 87.87 | 0.98 | 2.04 | 2.17 |
|         |                          | PhysGraph (No target) | 20.50 | 1.42 | 2.34 | 2.45 |
|         |                          | **DexFuture (Pred)** | **89.79** | **0.61** | **2.03** | **2.40** |
| 1292e@0 | pour / mug / mug | ManipTrans (GT) | 45.60 | 6.78 | 3.50 | 3.35 |
|         |                  | PhysGraph (GT) | 49.77 | 5.93 | 3.07 | 4.33 |
|         |                  | PhysGraph (No target) | 1.82 | 8.37 | 4.13 | 4.48 |
|         |                  | **DexFuture (Pred)** | **41.05** | **7.27** | **3.82** | **3.85** |
| 817fb@0 | wipe / big brush / whiteboard | ManipTrans (GT) | 50.05 | 1.55 | 2.17 | 2.24 |
|         |                               | PhysGraph (GT) | 62.24 | 1.33 | 1.54 | 1.86 |
|         |                               | PhysGraph (No target) | 8.99 | 2.77 | 2.68 | 2.58 |
|         |                               | **DexFuture (Pred)** | **56.96** | **1.50** | **2.42** | **2.43** |
| fc88d@0 | wipe / small brush / whiteboard | ManipTrans (GT) | 71.21 | 1.21 | 2.44 | 1.79 |
|         |                                 | PhysGraph (GT) | 80.65 | 1.11 | 2.49 | 1.50 |
|         |                                 | PhysGraph (No target) | 4.87 | 2.23 | 3.72 | 3.04 |
|         |                                 | **DexFuture (Pred)** | **67.17** | **1.25** | **3.02** | **1.70** |
| e1fa6@0 | shear / scissors / paper | ManipTrans (GT) | 15.16 | 1.18 | 3.31 | 2.25 |
|         |                          | PhysGraph (GT) | 35.84 | 1.35 | 2.52 | 1.83 |
|         |                          | PhysGraph (No target) | 0.00 | 1.78 | 3.70 | 2.68 |
|         |                          | **DexFuture (Pred)** | **30.69** | **1.62** | **2.97** | **1.92** |
| 9bb17@5 | shear / scissors / paper | ManipTrans (GT) | 51.68 | 1.93 | 2.26 | 2.86 |
|         |                          | PhysGraph (GT) | 59.21 | 1.90 | 3.12 | 2.70 |
|         |                          | PhysGraph (No target) | 11.18 | 3.85 | 3.81 | 3.93 |
|         |                          | **DexFuture (Pred)** | **48.66** | **2.49** | **2.73** | **2.90** |

**说明**:
- **DexFuture 平均 SR 59.69%**,达到 **PhysGraph-GT oracle (66.52%) 的 ~90%**;
- PhysGraph (No target) 平均仅 ~7%,凸显 target 条件对策略的关键性;
- 在 9fc3e@0（fruit knife cutting）DexFuture 甚至超过 oracle PhysGraph-GT;
- pour/mug 任务上所有方法表现都偏弱,该任务被作者归为最难的接触受限场景之一。

### Table 2: 未来 State 目标预测精度

| Task ID | 3D ↓ | UV ↓ | PCK@5 ↑ | PCK@10 ↑ |
|---------|-----:|-----:|--------:|---------:|
| 083f7@0 | 1.47 | 3.32 | 79.56 | 98.37 |
| 9fc3e@0 | 0.87 | 2.56 | 91.34 | 99.78 |
| 598a5@0 | 1.31 | 2.73 | 90.24 | 97.82 |
| 1292e@0 | 2.31 | 6.91 | 38.49 | 80.02 |
| 817fb@0 | 4.54 | 8.74 | 19.33 | 68.27 |
| e1fa67@0 | 5.54 | 25.58 | 7.98 | 32.93 |
| 9bb17@5 | 4.50 | 12.86 | 22.31 | 54.15 |
| cde36@1 | 1.65 | 5.41 | 76.93 | 95.76 |

**说明**: 切割/搅拌类任务（083f7、9fc3e、598a5、cde36）预测精度高;shear 类任务（e1fa67、9bb17）由于"窄接触区 + 突变运动"显著掉点 —— 这也是 limitations 的主因。

### Table 3: 未来预测 Horizon 消融

| 任务 | Horizon | 3D ↓ | UV ↓ | PCK@5 ↑ | PCK@10 ↑ |
|------|---------|-----:|-----:|--------:|---------:|
| 083f7@0 | h=24 | 1.66 | 4.05 | 71.87 | 97.54 |
| 083f7@0 | **h=16** | **1.47** | **3.32** | **79.56** | **98.37** |
| 083f7@0 | h=8  | 1.19 | 3.68 | 76.99 | 96.34 |
| 9fc3e@0 | h=24 | 1.11 | 2.95 | 86.66 | 99.27 |
| 9fc3e@0 | **h=16** | **0.87** | **2.56** | **91.34** | **99.78** |
| 9fc3e@0 | h=8  | 0.84 | 2.46 | 92.15 | 99.72 |
| 598a5@0 | h=24 | 1.77 | 3.97 | 73.56 | 97.46 |
| 598a5@0 | **h=16** | **1.31** | **2.73** | **90.24** | **97.82** |
| 598a5@0 | h=8  | 1.38 | 2.86 | 89.72 | 97.40 |

**关键发现**: $h=16$ 在"预测难度 vs 策略指导时长"之间最优,$h=24$ 在所有任务上明显退化,$h=8$ 在简单任务上略好但策略指导时长不够。

### Table 4: 与 [[DexWM]] + [[Cross-Entropy Method (CEM)|CEM]] 规划对比（任务 083f7@0）

| Method | 速度 (Hz) | SR (%) | $E_t$ (cm) | $E_j$ (cm) | $E_{xt}$ (cm) |
|--------|---------:|-------:|-----------:|-----------:|--------------:|
| DexWM (Finetune + H1) | 0.24 | 69.57 | 1.20 | 2.29 | 1.93 |
| DexWM (No-finetune + H1) | 0.24 | 0 | 2.94 | 3.65 | 3.41 |
| DexWM (Finetune + H16) | 0.24 | 0 | 2.82 | 2.78 | 2.55 |
| DexWM (No-finetune + H16) | 0.24 | 0 | 3.17 | 2.97 | 2.92 |
| **DexFuture (Pred target, H16)** | **60** | **83.49** | **1.06** | **2.08** | **2.04** |

**关键发现**: DexFuture 比 DexWM-CEM 快 ~250×,且无需 GT target;CEM 把 horizon 从 1 拉到 16 直接崩盘,说明 action-conditioned world model + CEM 在高维双手动作空间面对长 horizon 不可行。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[OakInk2]] | 多任务、bimanual hands × tool × object 接触轨迹 | 长时程、富接触双手工具使用 | 训练 + 评估 |

涉及子任务: cut（chop knife/bread, fruit knife/apple）、pour（mug/mug）、wipe（big/small brush/whiteboard）、shear（scissors/paper）、stir（隐含在 setup 中）。

### 实现细节

- **历史长度** $K$: 用于 visuomotor 输入
- **Horizon set** $\mathcal{H}$: $\{0,2,4,\ldots,16\}$,共 9 个 sparse target
- **低层控制频率**: 60 Hz
- **训练 backbone**: vision encoder + horizon-conditioned transformer（[[AdaLN|AdaLN]] block）
- **策略训练**: PPO,reward 见公式 13
- **评估硬件**: 单卡 RTX 3090Ti（与 DexWM 原配置 8×H100 形成鲜明对比）
- **DexWM baseline 适配**: state-based CEM（horizon=16, samples=128, elites=16, 4 iters）,~0.24 Hz

### 可视化结果

Figure 3 显示 DexFuture 在所有四种代表性任务上 rollout 与 oracle PhysGraph-GT 视觉相当;PhysGraph-NoTarget 频繁掉工具、失去接触;DexFuture 在 fruit knife 任务上甚至更稳。Figure 4 显示 CEM 仅在 H1+Finetune 配置下有可视稳定 rollout,其它三种皆崩。

---

## 批判性思考

### 优点

1. **target 预测 vs action 预测的范式转换**: 把"长 horizon 高维动作"问题转成"长 horizon 中等维度结构化 target"问题,显著降低预测难度,实证 250× 提速。
2. **horizon set + [[AdaLN|AdaLN]]**: 一个模型负责多 horizon 输出,避免训练 9 个独立 head 或自回归 rollout,效率高。
3. **结构化 token 复用 [[PhysGraph]]**: per-link 表示天然契合双手灵巧手的多链结构,且让低层策略保持 PhysGraph 的 inductive bias 收益。
4. **实验设计公平**: 对 oracle 同时给 GT,对 CEM 公平地做了 state-based 适配,且公开报告 0.24 Hz 这种"难看"的数字,可信度高。

### 局限性

1. **shear 类任务预测精度差**: e1fa67@0 PCK@5 仅 7.98,UV 误差 25.58 cm,说明窄接触 + 高动态运动场景下 visuomotor 历史不足以预测精确 target。
2. **仅在 simulation / [[OakInk2]] 上评估**: 未做真实双手机器人部署,sim-to-real gap 未验证。
3. **没有不确定性量化**: 高层 target 预测错误时低层只是盲目跟踪,缺乏 fallback / uncertainty-aware 策略。
4. **horizon 集合人工选择**: $\mathcal{H}=\{0,2,4,\ldots,16\}$ 是经验配置,不同任务可能需要不同 spacing。
5. **未对比直接 [[Action Chunking|action chunking]]**: 与 ACT-style chunk policy 没有直接对照,target chunking vs action chunking 哪个更优依旧悬而未决（虽然作者在 appendix 7.2 讨论了 relation,但缺定量对比）。

### 潜在改进方向

1. **uncertainty-aware target prediction**: 输出 target 的协方差,低层根据置信度调整 stiffness 或触发 replan。
2. **contact-aware tokenization**: 显式在 token 上预测/嵌入接触点位置（特别针对 shear / pour 场景）。
3. **加入触觉 / force 反馈**: 当前只用 RGB + 本体,接触失败时无法用力觉补救。
4. **real-robot 部署**: OakInk2 的 demonstration 来自动捕,sim2real 需要 domain randomization 或 force-based fine-tune。
5. **可学习 horizon spacing**: 让 $\mathcal{H}$ 也成为可学参数或在 inference 时自适应密度。

### 可复现性评估
- [ ] 代码开源（论文未声明 release 仓库）
- [ ] 预训练模型
- [ ] 训练细节完整（appendix 给了实现细节、PPO reward、训练流程）
- [x] 数据集可获取（[[OakInk2]] 公开）

---

## 关联笔记

### 基于
- [[PhysGraph]]: 低层 per-link transformer 策略架构直接借鉴
- [[ManipTrans]]: 早期"GT target → policy"路线,作为评估 baseline
- [[OakInk2]]: 数据集与评估协议

### 对比
- [[DexWM]]: action-conditioned world model + [[Cross-Entropy Method (CEM)|CEM]] 规划路线,作为 Sec 4.5 主要对照
- [[Action Chunking|ACT 风格 action chunking]]: 直接预测低层 action chunk 的另一条路线（appendix 讨论）

### 方法相关
- [[Future-State Visuomotor Target Predictor]]: 核心 contribution
- [[Horizon-Conditioned Transformer]]: 关键 backbone
- [[AdaLN]]: horizon 条件注入机制
- [[FiLM Modulation]]: per-link token 调制
- [[Receding-Horizon Execution]]: 高低层频率解耦执行策略
- [[Target-Conditioned Structured Dexterous Policy]]: 低层策略架构
- [[Cross-Entropy Method (CEM)]]: 对比 baseline 中的规划方法

### 数据集/硬件
- [[OakInk2]]: 训练与评估数据集
- [[Bimanual Dexterous Manipulation]]: 任务类别

---

## 速查卡片

> [!summary] DexFuture (arXiv 2606.05699, 2026-06)
> - **核心**: hierarchical 预测未来 target trajectory 替代 demo 特权 target + 慢 CEM 规划
> - **方法**: 高层 horizon-conditioned target predictor（[[AdaLN|AdaLN]] + 结构化 visuomotor token） + 低层 PhysGraph 风格 target-conditioned policy + 60 Hz receding-horizon 执行
> - **结果**: [[OakInk2]] 上达 90% PhysGraph-GT oracle 性能、~250× 快于 DexWM-CEM,fruit knife 任务甚至超过 oracle
> - **代码**: 论文未公开仓库

---

*笔记创建时间: 2026-06-08 07:32*
