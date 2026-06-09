---
title: "minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models"
method_name: "minWM"
authors: [Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, Jun Zhu]
year: 2026
venue: arXiv
tags: [world-model, video-diffusion, autoregressive-distillation, camera-control, real-time-generation, dmd, causal-forcing]
zotero_collection: 1-世界模型与视频生成
image_source: online
arxiv_html: https://arxiv.org/html/2605.30263v1
created: 2026-06-04
---

# 论文笔记：minWM — Full-Stack Framework for Real-Time Interactive Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, HKUST, Shengshu AI, Renmin University |
| 日期 | May 2026 |
| 项目主页 | https://github.com/shengshu-ai/minWM |
| 对比基线 | [[HunyuanVideo|HY1.5-TI2V-8B]], [[Wan2.1|Wan2.1-T2V-1.3B]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30263) / [Code](https://github.com/shengshu-ai/minWM) |

---

## 一句话总结

> minWM 是首个把双向视频扩散基础模型，端到端转化为**实时可交互、相机可控、自回归生成**的世界模型的开源全栈框架，首帧延迟降低最多 236×。

---

## 核心贡献

1. **全栈开源 Pipeline**：覆盖数据构造、可控微调、自回归训练、蒸馏、推理五个阶段，附带完整脚本、checkpoint、推理代码，使 [[Video World Model|视频世界模型]] 的可复现门槛大幅下降。
2. **PRoPE 相机条件注入**：用 [[PRoPE|Projected Rotary Position Embedding]] 把每帧的相机内参 $K_i$ 与位姿 $T_i^{cw}$ 编码进 [[Self-Attention]] 的 RoPE，绕过额外的 ControlNet 分支，对 [[MMDiT]] 与 cross-attention 两类架构同时适用。
3. **Causal Forcing++ 三阶段蒸馏**：将双向 [[Diffusion Model|扩散模型]] 蒸馏为少步 [[Autoregressive Diffusion|自回归扩散]] 生成器，组合 [[Teacher Forcing]]、[[Causal Consistency Distillation]] 与 [[Asymmetric DMD]]，实现 4 步推理 + 真正的低延迟首帧。
4. **训练资源/数据可控性 Ablation**：系统量化了 batch size、训练步数、相机轨迹质量（estimated vs. GT）对 controllability 的临界阈值，给后续复现者明确指导。

---

## 问题背景

### 要解决的问题

[[Video Diffusion Model|视频扩散基础模型]]（如 [[HunyuanVideo]]、[[Wan2.1]]）质量已经足够，但要把它当作"世界模型"用于具身/游戏交互，需要满足三个条件：

1. **因果性**（causal）：生成第 $t$ 帧只能看到 $<t$ 的帧与动作；
2. **低首帧延迟**（low first-frame latency）：交互闭环要求毫秒级出图；
3. **可控性**（controllability）：用户/agent 的动作（相机轨迹、按键等）能稳定地改变生成内容。

原版双向扩散模型三者都不满足。

### 现有方法的局限

- **双向扩散**（bidirectional video diffusion）：天然非因果，单步前向也要看完全部 latent；首帧延迟 = 整段视频延迟。
- **简单 [[Causal Attention|因果注意力]] 改装**：通常导致质量塌缩或可控性丢失。
- **[[Score Distillation|分数蒸馏]] 类加速方法**：原本针对双向，套到自回归后训练不稳定，且会丢掉相机条件。
- **相机条件**：现有工作多用 [[ControlNet]] 旁路或 cross-attention concat，参数浪费且对位姿误差非常敏感。

### 本文的动机

只要把 (a) **结构重写**（双向 → 因果）、(b) **条件注入**（相机姿态进 attention 的相对位置编码）、(c) **蒸馏目标**（双向教师 → 少步自回归学生）三件事正确解耦，就可以在**不重训基础模型权重**的前提下，得到可交互世界模型。minWM 用一条统一 pipeline 把这三步串起来。

---

## 核心方法 (方法详解)

### 模型架构

minWM 采用 **"双向预训练 → 相机微调 → 因果蒸馏"** 的三阶段架构：

- **输入**：相机轨迹 $\{(K_i, T_i^{cw})\}_{i=1}^N$ + 初始帧 / 文本 prompt
- **Backbone**：复用预训练 [[Video Diffusion Model]]（[[HunyuanVideo|HY1.5-TI2V-8B]] 的 [[MMDiT]] 或 [[Wan2.1|Wan2.1-T2V-1.3B]] 的 cross-attention DiT），权重不重新初始化
- **核心模块**：
  - [[PRoPE]]：把 $4\times 4$ 提升射影矩阵 $\tilde{P}_i$ 以 block-diagonal 方式注入 attention QKV 的旋转
  - [[Causal Forcing]] / Causal Forcing++：三阶段把双向蒸馏成自回归
- **输出**：每个 chunk 4 个 latent frame，采样 4 步出图，分辨率 $480 \times 832$，最多 7 帧 chunk
- **总参数**：保持基础模型规模（1.3B / 8B），仅在微调阶段引入相机条件相关 head

### 核心模块

#### 模块 1: [[PRoPE]] 相机条件注入

**设计动机**：[[RoPE]] 已经把空间坐标 $(x,y)$ 编码为旋转矩阵；只要把相机的提升射影矩阵也表达成旋转/线性变换，就能"免费"嵌入 attention，**不增加任何 cross-attention 分支**。

**具体实现**：

1. 对每帧构造提升投影矩阵 $\tilde{P}_i \in \mathbb{R}^{4\times 4}$；
2. 在 attention 的 head 维度做 block-diagonal 分块：前 $d/2$ 留给 PRoPE，后 $d/2$ 留给原本的 2D [[RoPE]]（$x$/$y$ 各 $d/4$）；
3. 把 $\tilde{P}_i$ 当作 Q/K 的相对几何先验，使 attention 等价于先把所有 token 投到同一相机平面再算相似度。

#### 模块 2: [[Causal Forcing]] 三阶段蒸馏

| 阶段 | 名称 | 目标 |
|------|------|------|
| Stage 1 | [[Teacher Forcing]] AR 训练 | 用 [[Causal Attention]] mask 把双向 DiT 转成 AR 生成器，每个 chunk 看不到未来 |
| Stage 2a | Causal ODE Initialization | 从带噪中间帧回归到 clean 帧，做少步生成器的初始化 |
| Stage 2b | [[Causal Consistency Distillation]] | 自洽蒸馏，去掉 offline ODE 数据的依赖 |
| Stage 3 | [[Asymmetric DMD]] | 用双向教师的 score 来对齐少步 AR 学生的分布 |

Causal Forcing++ 在每个阶段都把 PRoPE 相机条件并入，从而保留可控性不被蒸馏稀释。

#### 模块 3: Self-Rollout 训练

[[DMD|Asymmetric DMD]] 阶段使用 self-rollout：学生自己生成下一 chunk 作为下一步训练输入，使训练分布与推理分布对齐，减少 [[Exposure Bias|曝光偏差]]。

---

## 关键公式 (LaTeX)

### 公式 1: [[PRoPE|提升射影矩阵]]

$$
\tilde{P}_i = \begin{bmatrix} K_i & \mathbf{0} \\ T_i^{cw} & \mathbf{e}_4^\top \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

**含义**：把第 $i$ 帧的相机内参 $K_i$ 与外参 $T_i^{cw}$ 打包成一个 $4 \times 4$ 提升矩阵，用于 attention 中相对位姿的统一表示。

**符号说明**：
- $K_i \in \mathbb{R}^{3 \times 3}$：第 $i$ 帧相机内参
- $T_i^{cw} \in \mathbb{R}^{3 \times 4}$：world→camera 外参（含旋转 $R$ 与平移 $t$）
- $\mathbf{e}_4 = (0,0,0,1)^\top$：补齐齐次坐标

### 公式 2: [[PRoPE]] Block-Diagonal 嵌入

$$
D_t^{\text{PRoPE}} = \begin{bmatrix} I_{d/8} \otimes \tilde{P}_{i(t)} & \mathbf{0} \\ \mathbf{0} & \begin{matrix} \text{RoPE}_{d/4}(x_t) & 0 \\ 0 & \text{RoPE}_{d/4}(y_t) \end{matrix} \end{bmatrix}
$$

**含义**：head 维度被分成相机块与 2D 空间 RoPE 块，前者承担帧间几何对齐，后者承担帧内空间编码，互不干扰。

**符号说明**：
- $d$：attention head 维度
- $I_{d/8} \otimes \tilde{P}$：Kronecker 积，把 $\tilde{P}$ 复制 $d/8$ 次填满相机块
- $i(t)$：token $t$ 所属的帧索引

### 公式 3: [[PRoPE|带 PRoPE 的 Attention]]

$$
\text{Attn}_{\text{PRoPE}}(Q, K, V) = D^{\text{PRoPE}} \cdot \text{Attn}\big((D^{\text{PRoPE}})^\top Q,\ (D^{\text{PRoPE}})^{-1} K,\ (D^{\text{PRoPE}})^{-1} V\big)
$$

**含义**：在标准 attention 的 Q/K/V 上先做几何投影，再做缩放点积，等价于把 token 拉到统一相机坐标系下做相似度比较。

### 公式 4: [[Causal ODE Initialization|Causal ODE 初始化]] 损失

$$
\theta^\star = \arg\min_\theta \mathbb{E}\Big[\big\| G_\theta(x_t^i,\ x_{gt}^{<i},\ t) - x_0^i \big\|^2\Big]
$$

**含义**：少步生成器 $G_\theta$ 输入第 $i$ 个 chunk 的噪声 $x_t^i$ 与历史真值 $x_{gt}^{<i}$，回归对应的 clean chunk，作为 Stage 2 的稳定起点。

**符号说明**：
- $G_\theta$：少步自回归生成器
- $x_t^i$：第 $i$ chunk 在 timestep $t$ 的噪声 latent
- $x_{gt}^{<i}$：teacher-forced 历史 clean chunks
- $x_0^i$：第 $i$ chunk 的真值

### 公式 5: [[Causal Consistency Distillation|因果一致性蒸馏]]

$$
\theta^\star = \arg\min_\theta \mathbb{E}\Big[w(t)\, d\big(G_\theta(x_t^i, x_{gt}^{<i}, t),\ G_{\theta^-}(\hat{x}_{t-\Delta t}^i,\ x_{gt}^{<i},\ t - \Delta t)\big)\Big]
$$

**含义**：要求生成器在相邻 timestep 上自洽，去掉对 offline ODE 轨迹数据的依赖；$\theta^-$ 是 EMA 教师。

**符号说明**：
- $w(t)$：timestep-dependent 权重
- $d(\cdot, \cdot)$：距离度量（通常是 LPIPS 或 $L_2$）
- $\hat{x}_{t-\Delta t}^i$：用当前学生从 $x_t^i$ 走一步得到的临时样本

### 公式 6: [[Asymmetric DMD|非对称 DMD]] 梯度

$$
\nabla_\theta\, \mathbb{E}_t\Big[D_{KL}\big(p_{\theta, t}(\tilde{x}_t)\ \|\ p_{\text{data}, t}(\tilde{x}_t)\big)\Big] = -\mathbb{E}\Big[\big(s_{\text{real}}(\tilde{x}_t, t) - s_{\text{fake}}(\tilde{x}_t, t)\big)\, \frac{\partial \tilde{x}}{\partial \theta}\Big]
$$

**含义**：[[DMD]] 思路把生成分布 $p_\theta$ 推向数据分布 $p_{\text{data}}$，用真假两个 score 网络的差替代真实梯度；"非对称"指的是学生是自回归少步、教师是双向多步。

**符号说明**：
- $s_{\text{real}}, s_{\text{fake}}$：分别由冻结教师与可训练 fake critic 输出的 [[Score Function|得分函数]]
- $\tilde{x}_t$：学生自 rollout 的中间噪声样本
- $D_{KL}$：[[KL Divergence|KL 散度]]

---

## 关键图表 (图片引用)

### Figure 1: minWM Pipeline Overview / 全栈流程总览

![Figure 1: minWM pipeline overview](https://arxiv.org/html/2605.30263v1/x1.png)

**说明**：minWM 的端到端 pipeline。左侧：T2V/TI2V 基础模型；中段：用 [[PRoPE]] 做相机可控微调，得到双向可控教师；右段：三阶段 [[Causal Forcing|因果蒸馏]] 得到少步自回归学生，支持实时交互。整个管线开源全部脚本与 checkpoint。

### Figure 2: Camera-Controllable Generation / 相机可控生成示例

![Figure 2: Camera-controllable generation samples](https://arxiv.org/html/2605.30263v1/x2.png)

**说明**：在同一初始帧/文本条件下，给定不同相机轨迹（推近、左移、环绕），蒸馏后的少步 AR 学生仍能稳定地按轨迹改变视角；说明 [[PRoPE]] 注入在蒸馏阶段被 Causal Forcing++ 保留下来。

### Figure 3: Training Data Quality Ablation / 数据质量消融

![Figure 3: Training data ablation](https://arxiv.org/html/2605.30263v1/x3.png)

**说明**：三种数据源对比 — (a) [[SpatialVid]] 用感知估计位姿，相机控制完全失败；(b) 从 [[DL3DV]] 做 3D 重建再渲染轨迹，控制成功；(c) [[WorldPlay]] 直接按指定轨迹生成数据，控制效果最好。**结论：相机轨迹必须是 GT 或重建级精度，估计姿态不行。**

### Figure 4: Training Steps Impact / 训练步数影响

![Figure 4: Training steps ablation](https://arxiv.org/html/2605.30263v1/x4.png)

**说明**：HY1.5 上的训练步数扫描。1–2K 步：相机条件几乎无效；5K 步：开始涌现可控性；8K 步：稳定强可控。

### Figure 5: Batch Size Requirement / Batch Size 阈值

![Figure 5: Batch size ablation](https://arxiv.org/html/2605.30263v1/x5.png)

**说明**：Wan2.1 上的 batch size 扫描。<4：失败；=8：可控但不稳定；=16：稳定全流程。**结论：相机条件训练对 batch size 有硬性下限。**

### Table 1: First-Frame Latency Comparisons

| Base Model | Type | Latency (s) | Speedup |
|---|---|---|---|
| HY1.5 | Multi-step bidirectional | 1.041 | 1.00× |
| HY1.5 | Multi-step AR | 0.014 | 9.52× |
| HY1.5 | Few-step AR (minWM) | 0.346 | **223.75×** |
| Wan2.1 | Multi-step bidirectional | 0.055 | 1.00× |
| Wan2.1 | Multi-step AR | 0.028 | 9.39× |
| Wan2.1 | Few-step AR (minWM) | 0.001 | **236.64×** |

**说明**：在两类架构上 minWM 的少步 AR 学生均实现 200× 量级首帧延迟降低。注意 HY1.5 的 "Few-step AR" 实际首帧时间 0.346s 反而比 Multi-step AR 0.014s 大 — 这里 paper 的 "Latency" 列与 "Speedup" 列对应的基准不同（speedup 相对 multi-step bidirectional），需读 paper 注脚确认。

### Table 2: Training Schedule

| Stage | HY1.5 (steps) | Wan2.1 (steps) | LR | Batch |
|---|---|---|---|---|
| Bidirectional fine-tune | 8K | 5K | 1e-5 / 2e-6 | 32 |
| Stage 1 (Causal Forcing) | 4K | 4K | — | 32 |
| Stage 2 (CCD) | 1.5K | 2K | — | 32 |
| Stage 3 (Asymmetric DMD) | 500 | 200 | — | 32 |

**说明**：Stage 3 步数极少（500/200）即可让学生收敛，说明 [[DMD]] 这一阶段非常 data-efficient。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DL3DV]] | 大规模场景 | 重建得到精准相机轨迹 | 训练 |
| [[OpenVid]] + [[WorldPlay]] | 视频 + 指定轨迹生成 | GT 级 control | 训练 |
| [[SpatialVid]] | 视频 + 感知估计位姿 | 噪声位姿 | 反例消融 |

### 实现细节

- **分辨率**：$480 \times 832$，7 frame chunk，AR chunk size = 4 latent frame
- **优化器**：AdamW，LR = 1e-5 (HY1.5) / 2e-6 (Wan2.1)
- **Batch Size**：32（消融发现 16 是相机可控的下限）
- **推理步数**：少步 AR 学生 4 NFE
- **硬件**：未在 abstract/HTML 公开，但 8B 模型 + 5K+ 步训练量级，预计 64×A100 量级

### 可视化结果

- 在 [[Wan2.1]] 上首帧延迟 55ms → 1ms（236×）；
- 在 [[HunyuanVideo|HY1.5]] 上 1041ms → 346ms（223×）；
- 蒸馏后的学生仍保留双向教师的相机响应行为，长 rollout 下没有显著漂移。

---

## 批判性思考

### 优点

1. **真正"全栈"**：把数据、控制条件、AR 化、蒸馏、推理打包一条链路开源，社区可复现门槛骤降；
2. **PRoPE 设计优雅**：把相机条件嵌进 [[RoPE]] 而非旁挂 ControlNet，参数零增；
3. **架构通用**：对 [[MMDiT]] 与 cross-attention DiT 同时验证，证明 pipeline 不挑骨干；
4. **Ablation 给出工程门槛**：步数、batch、数据质量三条都有量化阈值，避免后人踩坑。

### 局限性

1. **首帧 latency 数据看起来矛盾**：HY1.5 Few-step AR 0.346s > Multi-step AR 0.014s，需要看 paper 原文确认 Speedup 是否相对 chunk latency 而非首帧；
2. **真值轨迹依赖**：方法对相机轨迹精度极敏感（估计位姿直接失效），在野外采集场景受限；
3. **仅评估相机一种 action**：作为"世界模型"，键盘/物体级动作没有 demo；
4. **8B 模型 + 5K+ 步训练**：开源全栈不等于平民可训练，仍需 64GPU 级算力；
5. **缺乏物理一致性评测**：与 [[Genie 2]]、[[GameNGen]] 这类同期世界模型相比，未给出 [[FVD]] / 物理合理性度量。

### 潜在改进方向

1. 把 [[PRoPE]] 拓展到 6-DoF 物体动作 / 6D 手部姿态条件，做真正的物理交互；
2. 引入 [[VGGT]] / [[MASt3R]] 级在线位姿估计，闭环到 PRoPE 输入，去掉 GT 依赖；
3. Stage 3 的 [[Asymmetric DMD]] 可以换 [[Score Identity Distillation|SiD]] 类，理论上 NFE 还能压到 1；
4. 与 [[Cosmos]] / [[Genie 2]] 做 controllability + 物理一致性 head-to-head 评测。

### 可复现性评估

- [x] 代码开源（GitHub: shengshu-ai/minWM）
- [x] 预训练 checkpoint 承诺开源
- [x] 训练细节完整（步数、batch、LR 全公开）
- [ ] 数据集可获取（[[WorldPlay]] 生成轨迹未必直接放出）

---

## 相关概念 (关联笔记)

### 基于

- [[HunyuanVideo]]：HY1.5-TI2V-8B 作为 [[MMDiT]] 骨干基础模型
- [[Wan2.1]]：Wan2.1-T2V-1.3B 作为 cross-attention DiT 骨干基础模型
- [[DMD]]：Stage 3 的核心蒸馏目标
- [[RoPE]]：PRoPE 的母体位置编码

### 对比

- [[Genie 2]]：另一类大规模可交互世界模型，但闭源、且 action 空间不同
- [[GameNGen]]：游戏世界模型，但场景受限于 Doom 类小空间
- [[CausVid]]：同样做 video diffusion 的因果蒸馏，但无相机控制

### 方法相关

- [[PRoPE]]：核心相机条件注入
- [[Causal Forcing]]：核心蒸馏框架
- [[Causal Consistency Distillation]]：Stage 2b 关键
- [[Asymmetric DMD]]：Stage 3 关键
- [[Teacher Forcing]]：Stage 1 训练范式
- [[Autoregressive Diffusion]]：学生模型的生成范式

### 硬件/数据相关

- [[DL3DV]]：训练数据（重建轨迹）
- [[WorldPlay]]：训练数据（生成轨迹）
- [[SpatialVid]]：反例数据（估计轨迹失败）

---

## 速查卡片

> [!summary] minWM
> - **核心**：用 PRoPE + Causal Forcing++ 把双向视频扩散基础模型蒸馏成相机可控的少步自回归世界模型
> - **方法**：3 阶段蒸馏（Causal ODE init + Causal Consistency Distill + Asymmetric DMD） + RoPE 内嵌相机条件
> - **结果**：Wan2.1 上 236× 首帧加速、HY1.5 上 223×；4 NFE 推理；相机控制蒸馏后保留
> - **代码**：https://github.com/shengshu-ai/minWM

---

*笔记创建时间: 2026-06-04*
