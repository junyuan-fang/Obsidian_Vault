---
title: "LARA: Latent Action Representation Alignment for Vision-Language-Action Models"
method_name: "LARA"
authors: [Mengya Liu, Baoxiong Jia, Jiangyong Huang, Jingze Zhang, Siyuan Huang]
year: 2026
venue: ICML 2026
tags: [vla, latent-action-model, representation-alignment, diffusion-policy, flow-matching, open-x-embodiment, humanoid]
zotero_collection: 5-具身智能与机器人/2-人形机器人
image_source: online
arxiv_html: https://arxiv.org/html/2606.07100v1
created: 2026-06-09
---

# 论文笔记：LARA: Latent Action Representation Alignment for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | BIGAI（北京通用人工智能研究院）+ 清华 / 北大合作 |
| 日期 | June 2026 (arXiv v1: 2026-06-05) |
| 项目主页 | https://lmy1001.github.io/ICML26_LARA/ |
| 代码 | https://github.com/lmy1001/LARA |
| 对比基线 | [[Pi05|π₀.₅]] / π₀-FAST / [[VLA|OpenVLA]] / GR00T-N1.6 / villa-X / UniVLA / SpatialVLA / CoT-VLA |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07100) / [HTML](https://arxiv.org/html/2606.07100v1) |

---

## 一句话总结

> 用 [[Latent Action Model|LAM]] 的"在线连续潜动作"和 [[VLA]] 的 [[DiT]] 中间层做 cosine 对齐，联合训练而非作 pseudo-label，把 LAM 的视觉动力学先验和 VLA 的动作监督互相正则化，在 [[Open X-Embodiment|OXE]] 受限预训练设定下 LIBERO 平均 88.6% / SIMPLER 65.2%，G1 真机 +32.1%。

---

## 核心贡献

1. **从 pseudo-label 到 representation alignment**: 跳出"LAM 离散 token 当作 VLA 监督"的传统做法，把 LAM 当作 *online teacher* 与 VLA 主干联合优化，alignment loss 直接走 DiT 中间表征。
2. **双向正则化**: LAM 的 forward-dynamics 给 VLA 注入"未来 state evolution"先验，VLA 的 inverse-dynamics 监督反过来逼 LAM 只关心控制相关视觉变化（不再被光影 / 背景扰动主导）。
3. **三种使用模式**: full pre-training（OXE-Constrained 设定下从零 train）、post-training enhancement（接 GR00T-N1.6 / π₀.₅ 等现成 VLA 做轻量增强）、LAM refinement（反过来用 LARA 改善 LAM 自己的 latent 质量）。
4. **真机 SOTA**: Unitree G1 humanoid 双手两个组合任务（Pick-Place、Grasp-Pour）每任务 50 demo 上，OXE-constrained 设定从 54% → 74%（+32.1%）。

---

## 问题背景

### 要解决的问题

[[VLA]] 训练需要大量带 action 标签的真机轨迹，但这类数据稀缺；与此同时 [[Latent Action Model|LAM]] 能从大量无标注人类 / 机器人视频中学到 latent action 表示，传统做法是把 LAM 输出当作 pseudo-label 监督 VLA。该范式的 bottleneck 在于：

- LAM 是 **frozen 的**，预训练时学到什么视觉动力学就锁死什么，无法被下游动作监督修正；
- LAM 的 latent 空间没有被 inverse-dynamics 约束，容易把光影 / 镜头抖动等"虚假视觉变化"当成 action，污染 VLA 的训练信号；
- VLA 端缺乏对未来视觉演化的显式建模，[[Flow Matching]] action head 容易生成"功能上不可达"的轨迹（撞墙 / 穿模）。

### 现有方法的局限

| 范式 | 代表 | 局限 |
|------|------|------|
| Discrete LAM token 作 pseudo-label | LAPA / Moto-GPT / GR00T-N1 | LAM 冻结，token 离散损失大；下游动作监督无法回流改善 LAM |
| 单纯 [[VLA]] 直接 SFT | [[VLA\|OpenVLA]] / [[Pi05]] / π₀-FAST | 缺少视觉动力学先验，长程任务（LIBERO-Long）泛化差 |
| World Model + Action 联合 | DreamVLA / villa-X / [[World-Action Model\|WAM]] | 需显式生成像素级 future frame，训练昂贵 |

### 本文的动机

如果 LAM 和 VLA 都是"动作编码器"，能不能让它们的 *中间表征* 在隐空间直接对齐？这样 LAM 既不冻结（可被 VLA 的动作监督拉向控制相关），VLA 又能借 LAM 的 forward-dynamics 获得视觉演化先验，不需要解码到像素。这就是 **Representation Alignment**（图像生成里 REPA 的思路）在 VLA 上的迁移。

---

## 方法详解

### 模型架构

LARA 是 *plug-and-play* 框架，把 [[Latent Action Model|LAM]] 与 [[VLA]] 主干通过一条 alignment loss 接在一起：

- **输入**: 视觉观测 $I_t$、语言指令 $l$、本体感知状态 $s_t$、连续观测对 $(I_t, I_{t+C})$（用来训 LAM）
- **VLA 主干**: 冻结的 **Eagle-2 VLM**（with learnable adapters）→ vision-language feature $f_t^{vl}$ → **cross-attention DiT**（Diffusion Transformer，[[Flow Matching]] velocity field），embodiment-specific MLP 编码 proprio state
- **LAM**: 三件套
  - [[Inverse Dynamic Model|IDM]] $\Phi_\text{IDM}$: $(I_t, I_{t+C}) \to z_t$（continuous latent action）
  - **Vector Quantizer** (VQ-VAE codebook，128 token)：$z_t \to z_t^q$
  - [[Forward Dynamic Model|FDM]] $\Phi_\text{FDM}$: $(I_t, z_t^q) \to \hat I_{t+C}$（重建未来帧）
- **Alignment 接口**: DiT 第 $L-2$ 层中间表征 $h_t^\theta$ 通过 MLP projection head $f_\psi$，与 LAM **连续** latent $z_t^\phi$ 做 cosine 相似度对齐
- **输出**: [[Action Chunking|动作 chunk]] $A_t = (a_t, \dots, a_{t+H-1})$，长度 $H$ 由 embodiment 决定

### 核心模块

#### 模块 1: Latent Action Model (LAM) 三件套

**设计动机**: 用 [[Inverse Dynamic Model|IDM]] 从两帧观测抽出 *与动作相关* 的视觉变化，[[Forward Dynamic Model|FDM]] 强制 latent 必须能预测未来帧来防止退化，[[Vector Quantization|VQ-VAE]] 把连续 latent 离散化得到 token codebook（128 entries）。

**具体实现**:
- IDM：双 frame 输入卷积 + transformer 抽特征
- VQ：codebook 128 token，commitment loss 系数 $\beta = 0.25$
- FDM：以 $I_t$ + quantized latent $z_t^q$ 为条件预测 $\hat I_{t+C}$，重建用 L2 像素损失
- **关键**: LARA 训练时给 DiT 对齐用的是 *quantize 之前的连续 $z_t$*，不是 codebook index，避免离散化信息损失

#### 模块 2: Cross-attention DiT 主干（VLA 端）

**设计动机**: 冻结大 VLM 当语义编码器，可学习 adapter + 专门的 [[DiT]] 处理动作 chunk，[[Flow Matching]] velocity field 输出 $H$ 步动作。

**具体实现**:
- 冻结 Eagle-2 VLM 输出 $f_t^{vl}$（vision token + language token）
- DiT 接受 noised action $A_t^\tau$、conditioning $c_t = \{s_t, f_t^{vl}\}$
- DiT 拆为 encoder $E_\theta$（出中间表征 $h_t^\theta$）+ decoder $D_\theta$（输出 velocity $\hat v_t$）
- Alignment 接在第 $L-2$ 层（对 GR00T-N1.6 backbone 是最优深度，对 π₀.₅ 是 $L$ 层）

#### 模块 3: Representation Alignment（核心新意）

**设计动机**: 把 REPA（Representation Alignment for Diffusion）的范式从 image generation 搬到 VLA：教师不再是 frozen pretrained encoder，而是 *online* 学习的 LAM。

**与 REPA 的差异**：
- REPA：$y_t^\text{pretrain}$ 来自冻结 DINO / MAE 等 visual encoder
- LARA：$z_t^\phi$ 来自 **在线训练的** LAM，构成 *双向梯度流*——alignment loss 同时回传到 $\theta$（VLA）和 $\phi$（LAM）

#### 模块 4: 双向正则化机制

**Forward Dynamics Grounding for VLA**:
> 把 action-DiT 表征锚定到 LAM 的 forward-predictive latent 上，相当于给 [[Flow Matching]] velocity 灌入"动作执行后视觉会变成什么样"的隐式约束，减少功能上不可达的轨迹（撞墙、穿模、抓取失败）。

**Inverse Dynamics Regularization for LAM**:
> VLA 端的真实动作监督通过 alignment loss 回流到 LAM，逼 LAM 的 latent 只保留控制相关视觉变化，抑制光影 / 背景 / 镜头抖动等 spurious factor。这也是为什么 Table 3 显示 LARA-LAM 比原始 LAM 的 SIMPLER 平均成功率 +15.7%。

### LARA 训练 Pipeline（三阶段）

| 阶段 | 数据 | 损失 | 更新 | 目标 |
|------|------|------|------|------|
| Stage 1: LAM Pre-training | 无标注视频（机器人 + 人类视频） | $\mathcal{L}_{LAM}$ (Eq. 3) | $\phi$ | 通用 latent action codebook |
| Stage 2: LARA Joint Pre-training | OXE 带 action 标签数据 | $\mathcal{L} = \mathcal{L}_{ACT} + w_1\mathcal{L}_{LARA} + w_2\mathcal{L}_{LAM}$ | $\theta, \phi, \psi$ | 联合优化 VLA + LAM |
| Stage 3: LARA Joint Post-training | 目标任务 demo（如 LIBERO / G1-Real） | 同 Stage 2 | $\theta, \phi, \psi$ | embodiment 特化 |

**三种应用模式**：
- **Full Training**: 从零跑完三阶段（实验对应"OXE-Constrained"）
- **Post-training Enhancement**: 接现成 VLA（GR00T-N1.6 / π₀.₅）只跑 Stage 3（实验对应"Unconstrained"）
- **LAM Refinement**: 反过来把 LARA 训出的 LAM 用作下游 pseudo-label 源，喂给 Moto-GPT 风格 VLA（+15.7% on SIMPLER）

---

## 关键公式

### 公式 1: [[Flow Matching|Flow-Matching Action Loss]]

$$
\mathcal{L}_{ACT}(\theta) = \mathbb{E}_{\tau,\epsilon}\left[\left\| v_\theta(A_t^\tau, c_t) - (A_t - \epsilon) \right\|_2^2\right]
$$

**含义**: 标准 [[Flow Matching]] velocity-matching 损失，约束 DiT 在每个 flow timestep 输出从噪声指向真实动作的速度场。

**符号说明**:
- $v_\theta$: DiT 速度场网络
- $A_t^\tau = \tau A_t + (1-\tau)\epsilon$: 第 $\tau$ 步的 noised action chunk
- $c_t = \{s_t, f_t^{vl}\}$: 条件（proprio state + VLM 特征）
- $\tau \sim \mathcal{U}(0,1)$: flow timestep；$\epsilon \sim \mathcal{N}(0,I)$

### 公式 2: 推理时的 Flow Integration

$$
A_t^{\tau + 1/K} = A_t^\tau + \frac{1}{K} v_\theta(A^\tau, c_t)
$$

**含义**: Euler 法 K 步积分从 noise → action。

**符号说明**: $K$ 为积分步数（实验取 10）；$A_t^0 = \epsilon$ 初始噪声。

### 公式 3: [[Latent Action Model|LAM]] 训练损失（VQ-VAE 目标）

$$
\mathcal{L}_{LAM}(\phi) = \|I_{t+C} - \hat I_{t+C}\|_2^2 + \|\mathrm{sg}[z_t^q] - z_t\|_2^2 + \beta \|z_t^q - \mathrm{sg}[z_t]\|_2^2
$$

**含义**: 三项：FDM 像素重建 + codebook 损失 + commitment 损失，标准 [[Vector Quantization|VQ-VAE]] 目标。

**符号说明**:
- $I_{t+C}$: 未来 $C$ 步观测；$\hat I_{t+C} = \Phi_\text{FDM}(I_t, z_t^q)$
- $z_t = \Phi_\text{IDM}(I_t, I_{t+C})$: 连续 latent
- $z_t^q$: codebook 查表后的量化 latent
- $\mathrm{sg}[\cdot]$: stop-gradient
- $\beta = 0.25$: commitment 权重

### 公式 4: DiT 的 Encoder-Decoder 拆分

$$
h_t^\theta = E_\theta(A_t^\tau, c_t), \qquad \hat v_t = D_\theta(h_t^\theta, c_t)
$$

**含义**: 把 DiT 在第 $L-2$ 层切开，前半段 $E_\theta$ 输出供 alignment 用的中间表征 $h_t^\theta$，后半段 $D_\theta$ 继续算 velocity。

**符号说明**: $L$ 是 DiT 总层数；alignment 接的"深度"是 $L-2$（实验最优）。

### 公式 5: 标准 [[Representation Alignment|REPA]] 损失（对比基线）

$$
\mathcal{L}_{RA}(\theta, \psi) = -\mathbb{E}_{A_t, \epsilon, \tau}\left[\mathrm{CosSim}\left(y_t^\text{pretrain}, f_\psi(h_t^\theta)\right)\right]
$$

**含义**: REPA 原版——把 DiT 中间表征对齐到 *frozen* 预训练 encoder $y_t^\text{pretrain}$。

**符号说明**: $f_\psi$ 是 MLP projection head；$y_t^\text{pretrain}$ 来自 frozen DINO / MAE 等。

### 公式 6: [[Latent Action Representation Alignment|LARA Alignment Loss]]（核心创新）

$$
\mathcal{L}_{LARA}(\theta, \phi, \psi) = -\mathbb{E}_{A_t, \epsilon, \tau}\left[\mathrm{CosSim}\left(z_t^\phi, f_\psi(h_t^\theta)\right)\right]
$$

**含义**: 与公式 5 的关键差异：teacher 是 *online* 学习的 LAM 连续 latent $z_t^\phi$（不是 frozen，不是 quantized 后的 codebook token），梯度同时回传给 LAM 参数 $\phi$ 和 VLA 参数 $\theta$。

**符号说明**: $z_t^\phi = \Phi_\text{IDM}(I_t, I_{t+C}; \phi)$ 来自在线 IDM。

### 公式 7: 全局 LARA 训练目标

$$
\mathcal{L}(\theta, \phi, \psi) = \mathcal{L}_{ACT}(\theta) + w_1 \mathcal{L}_{LARA}(\theta, \phi, \psi) + w_2 \mathcal{L}_{LAM}(\phi)
$$

**含义**: 三项联合优化——动作监督（来自真机标签）+ 表征对齐（双向信息流）+ LAM 自监督（视觉动力学）。

**符号说明**: $w_1, w_2$ 是损失权重；实验中 $w_1, w_2$ 的最优组合在 Appendix B.2 ablation。

---

## 关键图表

### Figure 1: LARA Overview / 框架总览

![Figure 1](https://arxiv.org/html/2606.07100v1/x1.png)

**说明**: LARA 框架把"无标注视频数据"和"action-labeled 机器人数据"桥接起来。三种应用模式同时支持：full pre-training、post-training enhancement、LAM refinement。LAM 学视觉动力学，DiT 学动作生成，alignment 桥两边。

### Figure 2: LAM-based VLA 范式对比

![Figure 2](https://arxiv.org/html/2606.07100v1/x2.png)

**说明**: 左：传统做法把 LAM 当 *frozen pseudo-label* 源（[[Latent Action Model|LAM]] → discrete token → 监督 VLA），LAM 一旦预训练完就锁死。右：LARA 把 LAM 与 VLA *联合优化*，通过 alignment loss 维持双向梯度流。这是核心范式转变。

### Figure 3: 方法架构详图

![Figure 3](https://arxiv.org/html/2606.07100v1/x3.png)

**说明**: 三大块连成一张图——
- **左侧 LAM**: [[Inverse Dynamic Model|IDM]] $(I_t, I_{t+C}) \to z_t$，[[Forward Dynamic Model|FDM]] $(I_t, z_t^q) \to \hat I_{t+C}$
- **中间 VLA**: 冻结 Eagle-2 VLM + cross-attention DiT，输入 $(I_t, l, s_t)$ 输出 $A_t$
- **桥接**: $z_t^\phi$（LAM 连续 latent）与 DiT 第 $L-2$ 层 $h_t^\theta$ 通过 projection head $f_\psi$ 做 cosine alignment

### Figure 4: 任务可视化（GR1-Sim & G1-Real）

![Figure 4](https://arxiv.org/html/2606.07100v1/x4.png)

**说明**: 左侧 GR1-Sim-24 一个代表性双手任务（24 任务 / 任务 30 demo），右侧 G1-Real 两个真机任务——
- **Pick-Place**: 抓绿色番茄 → 放到绿色篮子
- **Grasp-Pour**: 双手抓瓶 → 倒水入杯（Grasp-L、Grasp-R、Pour 三个 sub-task）

### Figure 5: Attention Map 可视化（LAM vs LARA-LAM）

![Figure 5](https://arxiv.org/html/2606.07100v1/x5.png)

**说明**: 上：原始 LAM 的 latent action 对图像 patch 的注意力——散布在背景 / 物体上；下：LARA-LAM 的注意力——*高度集中在末端执行器 (end-effector) 周围*，红色越深 attention 越强。直接可视化证明：VLA 的动作监督通过 alignment 反向逼 LAM 关注控制相关区域。

### Figure 6: 消融实验 (LIBERO-Long)

![Figure 6](https://arxiv.org/html/2606.07100v1/x6.png)

**说明**: 在 LIBERO-Long 上扫两个维度——
- **Alignment 深度**: 测 $\{4, 8, L-2, L\}$，GR00T-N1.6 上 $L-2$ 最优，π₀.₅ 上 $L$ 最优（更靠近 action head 更好，但具体最优层与 backbone 相关）
- **Joint 优化 vs Frozen LAM**: 联合优化显著优于冻结 LAM，验证 *双向梯度流* 必不可少

### Table 1: LIBERO + SIMPLER-ENV 全套对比

| Method | LIBERO Spat. | LIBERO Obj. | LIBERO Goal | LIBERO Long | LIBERO Avg | SIMPLER Pick | SIMPLER Move | SIMPLER Drawer | SIMPLER Avg |
|--------|---|---|---|---|---|---|---|---|---|
| **OXE-Constrained** | | | | | | | | | |
| [[VLA\|OpenVLA]] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 16.3 | 46.2 | 35.6 | 32.7 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 | 17.0 | 4.2 | 22.7 | 14.6 |
| Moto-GPT | — | — | — | — | — | 74.0 | 60.4 | 43.1 | 61.4 |
| LAPA | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 | — | — | — | — |
| LARA (DiT-only) | 84.5 | 90.0 | 86.5 | 76.5 | 84.4 | 62.3 | 84.0 | 21.0 | 55.8 |
| **LARA (full)** | **88.0** | **92.0** | **88.5** | **86.0** | **88.6** | **82.3** | **83.7** | **29.5** | **65.2** |
| Δ vs DiT-only | +4.1% | +2.2% | +2.3% | +12.4% | +5.0% | +32.1% | -0.4% | +40.5% | +16.8% |
| **Unconstrained** | | | | | | | | | |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 | 88.0 | 72.7 | 41.8 | 70.7 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 | — | — | — | — |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 | 75.3 | 67.5 | 42.9 | 61.9 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 | — | — | — | — |
| TraceVLA | — | — | — | — | — | 45.0 | 63.8 | 63.1 | 57.3 |
| Magma | — | — | — | — | — | 75.0 | 53.0 | 58.9 | 62.3 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 | 98.7 | 75.0 | 59.3 | 77.7 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 | — | — | — | — |
| GR00T-N1.6 | 97.5 | 96.0 | 95.5 | 91.0 | 95.0 | 97.3 | 87.0 | 52.3 | 78.9 |
| **GR00T-N1.6 + LARA** | **96.5** | **97.5** | **96.0** | **92.5** | **95.6** | **98.0** | **89.0** | **52.8** | **79.9** |
| Δ Post-train | -1.0% | +1.6% | +0.5% | +1.6% | +0.6% | +0.7% | +2.3% | +1.0% | +1.3% |

**说明**:
- OXE 受限设定下，LARA full 在 LIBERO-Long 比 DiT-only baseline +12.4%（说明 LAM 灌的视觉动力学先验对长程任务最有用）；
- SIMPLER 上的 Pick / Drawer 是巨大提升（+32.1% / +40.5%），但 Move 略降 0.4%；
- 在 Unconstrained（接 GR00T-N1.6）已经 95% 起跳的情况下，post-train +1.3% 依然 consistently 正向，证明 LARA 是 *no-regret enhancement*。

### Table 2: GR1-Sim 双手 24 任务 + G1 真机

| Method | GR1-Sim Avg | G1 Pick-Place（sub）| G1 Pick Full | Grasp-L | Grasp-R | Pour（sub）| Pour Full | G1 Avg |
|--------|---|---|---|---|---|---|---|---|
| **OXE-Constrained** | | | | | | | | |
| LARA (DiT-only) | 6.4 | 74.0 | 78.4 | 58.0 | 58.0 | 78.0 | 93.1 | 54.0 |
| **LARA (full)** | **11.4** | **90.0** | **88.9** | **80.0** | **80.0** | **84.0** | **100.0** | **74.0** |
| Δ | +78.1% | +21.6% | +13.4% | +37.9% | +37.9% | +7.7% | +7.4% | +37.0% |
| **Unconstrained** | | | | | | | | |
| GR00T-N1.6 | 47.0 | 90.0 | 84.4 | 76.0 | 78.0 | 80.0 | 87.2 | 72.0 |
| **GR00T-N1.6 + LARA** | **48.5** | **92.0** | **91.3** | **84.0** | **86.0** | **76.0** | **94.4** | **76.0** |
| Δ Post-train | +3.2% | +2.2% | +8.2% | +10.5% | +10.3% | -4.0% | +7.2% | +5.6% |

**说明**:
- DiT-only 在 GR1-Sim 24 任务上只能拿 6.4%（双手任务难度极大，几乎 collapse），加 LARA 拉到 11.4%（+78%，但绝对值仍很低，说明 GR1-Sim 是远未饱和的 benchmark）；
- G1 真机 Pour 子任务里 Pour-sub 反而轻微下降（−4%），说明 LARA 对纯流体类任务的视觉动力学先验帮助不如离散接触任务大；
- Grasp-L / Grasp-R 同步 +10% 表明双手对称改进。

### Table 3: LARA-LAM 反哺下游 (SIMPLER)

| Method | Pick Obj. | Move Near | Open Drawer | Close Drawer | Pick Coke | Avg |
|--------|---|---|---|---|---|---|
| 原始 LAM | 36.3 | 61.0 | 25.7 | 38.0 | 53.0 | 42.8 |
| **LARA-LAM** | **41.0** | **63.7** | **29.3** | **53.7** | **59.7** | **49.5** |
| Δ | +12.9% | +4.4% | +14.0% | +41.3% | +12.6% | **+15.7%** |

**说明**: 把 LARA 训练过程中 *副产物* 的 LARA-LAM 拿去当下游 Moto-GPT 风格 VLA 的 pseudo-label 源，平均 +15.7%（Close Drawer +41.3% 是最大提升）。说明双向梯度流真的把 LAM 改造成了更"动作觉察"的视觉编码器，可独立复用。

---

## 实验

### 数据集

| 数据集 | 规模 | 用途 | 说明 |
|--------|------|------|------|
| [[Open X-Embodiment\|OXE]] | 2.4M 轨迹 / 200+ embodiment | Stage 2 预训练 | OXE-constrained 设定唯一允许的预训练源 |
| 无标注视频集 | 公开机器人/人类视频 | Stage 1 LAM 预训练 | |
| [[LIBERO]] | Spatial / Object / Goal / Long 4 套 | 仿真评测 | 单臂 manipulation 标准 |
| SIMPLER-ENV | Pick Coke / Move / Drawer | 仿真评测 | Sim2Real gap 标准基准 |
| GR1-Sim-24(30) | 24 双手任务，每任务 30 demo | 仿真评测 | Fourier GR1 humanoid 仿真 |
| G1-Real(50) | 2 复合任务，每任务 50 demo | 真机评测 | Unitree G1 humanoid 双手 |

### 实现细节

- **Backbone**: Eagle-2 VLM（frozen + learnable adapter），cross-attention DiT 作 action expert
- **LAM**: IDM + VQ-VAE codebook 128 + FDM，commitment $\beta = 0.25$，预测 horizon $C$
- **Alignment 接入层**: GR00T-N1.6 上 $L-2$ 层最优；π₀.₅ 上 $L$ 层最优
- **Action chunk 长度 $H$**: 与 embodiment 绑定（appendix B 详）
- **Flow Matching 推理步数**: $K = 10$
- **训练**: Stage 1 → 2 → 3 串行；Stage 3 任务特化 demo $\ll$ Stage 2

### 可视化结果

- Figure 5 attention map 是最有说服力的定性证据：LARA-LAM 的注意力高度集中在末端执行器，传统 LAM 散漫无章；
- G1 真机 Grasp-Pour 任务中，LARA full 把 Pour-Full 拉到 100%（DiT-only 93.1%），双手协同明显改进。

---

## 批判性思考

### 优点
1. **范式漂亮**：把 image generation 里成熟的 REPA 思路迁到 VLA，*online teacher* 取代 frozen teacher 是真正的概念升级，不是 trick 堆叠；
2. **三模式即插即用**：同一框架既能 from-scratch，也能给 SOTA VLA（GR00T-N1.6）做后增强，可复用性极高；
3. **LAM 副产物可独立交付**：Table 3 证明 LARA-LAM 单独拿出去给 Moto-GPT 风格 VLA 用也涨 15.7%，是 paper 之外的额外赠品；
4. **真机评测扎实**：Unitree G1 + 50 demo / 任务 + 50 trial 评测，比纯仿真有说服力。

### 局限性
1. **GR1-Sim 24 任务的绝对值只有 11.4%**：双手任务的 ceiling 仍很低，LARA 是相对提升 78% 而非绝对 SOTA；
2. **Pour 任务 −4%**：对流体类 / 接触不离散的任务，forward dynamics 先验帮助有限，甚至略损；
3. **alignment 深度的最优层 backbone 相关**：GR00T 上 $L-2$、π₀.₅ 上 $L$，没有 backbone-agnostic 的统一原则，迁移到新 VLA 需要重新 ablation；
4. **w₁, w₂ 损失权重敏感**：appendix 显示组合空间很大，工程调参成本不低；
5. **post-training 增益已经 marginal**：GR00T-N1.6 + LARA 在 LIBERO-Avg 仅 +0.6%，部分 sub-task 还下降（Spatial −1%），接近 saturation regime；
6. **VLM 端冻结**：Eagle-2 一直 frozen，没有探索 VLM 端的 alignment（可能是下一篇）。

### 潜在改进方向
1. **跨模态 alignment**: 把 LAM 的 latent 也对齐到 VLM 的 vision feature，三方联合优化；
2. **多 horizon LAM**: 当前 LAM 用固定 $C$ step 观测对，可扩展到多尺度时间窗；
3. **Pour 失败的根因分析**: 流体动力学是否需要专门的 FDM 变体；
4. **与 [[World-Action Model|WAM]] 范式融合**: WAM 直接生成 future frame，LARA 只对齐 latent，两者结合可能更稳。

### 可复现性评估
- [x] 代码开源（https://github.com/lmy1001/LARA）
- [x] 项目主页 + 真机视频公开
- [x] 训练细节较完整（三阶段 pipeline 清晰）
- [x] 主要 benchmark 公开（LIBERO / SIMPLER / OXE）
- [ ] G1 真机数据未公开（合理）
- [ ] 预训练权重发布状态待确认

---

## 关联笔记

### 基于
- [[VLA]]: 主框架范式
- [[Latent Action Model|LAM]]: 视觉动力学先验来源
- [[Flow Matching]]: action head 的核心机制
- [[Vector Quantization|VQ-VAE]]: LAM 的离散化方式
- [[Representation Alignment]]: 思路源头（image gen 里的 REPA）

### 对比
- [[Pi05|π₀.₅]]: 强 baseline，被用作 unconstrained post-training 母体之一
- [[VLA|OpenVLA]]: OXE-constrained 经典基线
- [[World-Action Model|WAM]]: 另一种"未来感知 VLA"，但需要像素级生成
- Moto-GPT / LAPA: 传统 LAM-as-pseudo-label 范式代表

### 方法相关
- [[Inverse Dynamic Model|IDM]]: LAM 的核心组件
- [[Forward Dynamic Model|FDM]]: LAM 的核心组件
- [[DiT]]: VLA 主干架构
- [[Action Chunking]]: 输出形式

### 硬件 / 数据相关
- [[Open X-Embodiment|OXE]]: 预训练数据
- [[LIBERO]]: 评测基准
- SIMPLER-ENV / GR1-Sim / G1-Real: 评测基准

---

## 速查卡片

> [!summary] LARA: Latent Action Representation Alignment for VLA
> - **核心**: 用 *在线学习* 的 LAM 连续 latent 对齐 DiT 中间表征，取代 frozen pseudo-label 范式
> - **方法**: $\mathcal{L} = \mathcal{L}_{ACT} + w_1 \mathcal{L}_{LARA} + w_2 \mathcal{L}_{LAM}$，三损失联合，双向梯度
> - **结果**: LIBERO-Long 86.0%（DiT-only +12.4%），G1 真机平均 74.0%（+37%），LARA-LAM 副产物给下游 +15.7%
> - **代码**: https://github.com/lmy1001/LARA  •  **页**: https://lmy1001.github.io/ICML26_LARA/

---

*笔记创建时间: 2026-06-09*
