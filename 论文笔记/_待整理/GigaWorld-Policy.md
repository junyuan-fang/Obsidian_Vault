---
title: "GigaWorld-Policy: An Efficient Action-Centered World-Action Model"
method_name: "GigaWorld-Policy"
authors: [Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, Min Cao, Peng Li, Qiuping Deng, Wenjun Mei, Xiaofeng Wang, Xinze Chen, Xinyu Zhou, Yang Wang, Yifan Chang, Yifan Li, Yukun Zhou, Yun Ye, Zhichao Liu, Zheng Zhu]
year: 2026
venue: arXiv
tags: [world-action-model, robot-policy, flow-matching, diffusion-transformer, visuomotor-policy, video-prediction, action-chunking]
zotero_collection: _待整理
image_source: online
arxiv_html: https://arxiv.org/html/2603.17240
created: 2026-04-24
---

# 论文笔记：GigaWorld-Policy: An Efficient Action-Centered World-Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | GigaAI |
| 日期 | March 2026 |
| 项目主页 | [GigaWorld-Policy](https://gigaai-research.github.io/GigaWorld-Policy/) |
| 对比基线 | [[Motus]], [[Pi05|π₀.₅]], [[X-VLA]], [[Cosmos-Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.17240) / [Code](https://github.com/open-gigaai/giga-world-policy) / [HuggingFace](https://huggingface.co/open-gigaai) |

---

## 一句话总结

> 提出以动作为中心的 [[WAM|世界-动作模型]]，训练时联合预测动作与未来视频，推理时可选地跳过视频生成，实现比 Motus 快 9 倍且真实世界成功率高 7% 的高效机器人策略。

---

## 核心贡献

1. **以动作为中心的 WAM 架构**: 通过 [[Causal Attention|因果注意力掩码]] 解耦动作预测与视频生成，推理时视频生成可选，大幅降低延迟
2. **三阶段预训练流水线**: 从网络视频基础模型到具身数据预训练再到任务微调，约 10,000 小时具身数据提供丰富先验
3. **高效推理与强性能**: 在 A100 上推理仅需 360ms（Motus 3231ms 的 1/9），真实世界平均成功率 0.83 超越所有基线

---

## 问题背景

### 要解决的问题

现有 [[WAM|世界-动作模型]] 在推理时需要生成完整的未来视频序列才能获得动作，引入大量延迟，难以实现实时闭环控制。如何在保留视频预测带来的训练信号增益的同时，避免推理时的视频生成开销？

### 现有方法的局限

- **VLM-based VLA**（如 [[VLA]] + 辅助视频监督）：受限于 VLM 的判别性本质，难以充分建模视觉动态
- **联合动作-视频预测**（如 [[Motus]]）：使用双向注意力，推理时必须生成未来视频，延迟高达 3231ms
- **两阶段方法**（先生成视频再用逆动力学模型解码动作）：仍需完整视频预测，额外开销不可避免
- **整体问题**: 这些方法将视频生成与动作预测紧密耦合，推理效率低，限制了实时部署

### 本文的动机

如果将未来视频预测视为训练时的辅助监督信号（而非推理时的必要步骤），就可以在训练中利用视觉动态的丰富监督来提升动作质量，同时在推理时直接产生动作，跳过昂贵的视频生成。关键设计是 **因果注意力掩码**：动作 token 不依赖未来视频 token，从而实现推理时的解耦。

---

## 方法详解

### 模型架构

GigaWorld-Policy 采用基于预训练视频生成 [[Diffusion Transformer|扩散 Transformer]] 的架构：

- **输入**: 自然语言指令 $l$ + 多视角 RGB 观测 $o_t = \{o_t^{left}, o_t^{front}, o_t^{right}\}$ + 本体感受状态 $s_t$
- **Backbone**: 5B 参数的 [[Diffusion Transformer]] 视频生成模型（网络视频预训练）
- **核心模块**: [[Causal Attention|块级因果注意力掩码]] 解耦动作与视频预测
- **输出**: [[Action Chunking|动作块]] $a_{t:t+p-1}$ + 可选的未来视频帧
- **总参数**: 5B

### 核心模块

#### 模块 1: 多视角合成输入

**设计动机**: 利用多视角信息提供更完整的空间理解，同时保持高效的 token 化

**具体实现**:
- 将三个相机视角（左、前、右）拼接为单张合成图像

$$
o_t^{comp} = \text{Compose}(o_t^{left}, o_t^{front}, o_t^{right})
$$

- 使用预训练 [[VAE]] 编码为视觉 token $T_o$，保留空间结构并促进跨视角一致性
- 视觉 token 使用 2D 位置编码，状态和动作 token 使用 1D 时序位置编码

#### 模块 2: 块级因果注意力掩码

**设计动机**: 确保动作预测不依赖未来视频 token，从而在推理时可以跳过视频生成

**具体实现**:
- 所有模态的 token 拼接为统一序列：

$$
T_t = [T_o \;;\; T_s \;;\; T_a \;;\; T_f]
$$

- 注意力规则（见 Figure 4）：
  - **状态/观测 token** ($T_s$, $T_o$): 互相注意
  - **动作 token** ($T_a$): 仅注意 $\{T_s, T_o\}$，**不能看到** $T_f$
  - **未来视频 token** ($T_f$): 注意 $\{T_s, T_o, T_a\}$，可利用预测的动作
- 语言指令通过 [[Cross-Attention|交叉注意力]] 注入，不受因果掩码约束
- 所有 token 共享相同的 Q/K/V 投影矩阵，紧密耦合动作与视觉证据

#### 模块 3: 稀疏未来帧预测

**设计动机**: 逐帧预测冗余且昂贵，稀疏预测保留场景关键演化同时降低监督冗余

**具体实现**:
- 每隔 $\Delta$ 步预测一个未来观测帧
- 预测帧数 $K = \lfloor p / \Delta \rfloor$
- 最优间隔 $\Delta = 12$（消融实验验证）

---

## 关键公式

### 公式 1: [[VLA|VLA 策略分布]]

$$
a_{t:t+p-1} \sim q_\Theta(\cdot \mid o_t, s_t, l)
$$

**含义**: 标准 VLA 策略将语言指令、观测和状态作为条件，输出动作块

**符号说明**:
- $a_{t:t+p-1}$: 长度为 $p$ 的动作块
- $q_\Theta$: 参数为 $\Theta$ 的策略分布
- $o_t$: 多视角 RGB 观测
- $s_t$: 本体感受状态
- $l$: 自然语言指令

### 公式 2: [[WAM|统一世界-动作采样]]

$$
(a_{t:t+p-1},\; c_t) \sim g_\Theta(\cdot \mid o_t, s_t, l)
$$

**含义**: 统一模型同时采样动作块和用于视觉预测的隐式条件信号

**符号说明**:
- $c_t$: 动作隐式条件信号，引导未来视觉预测
- $g_\Theta$: 统一的世界-动作模型

### 公式 3: [[Video Prediction|未来视觉动态预测]]

$$
(o_{t+\Delta},\; o_{t+2\Delta},\; \ldots,\; o_{t+K\Delta}) \sim g_\Theta(\cdot \mid o_t, s_t, l, c_t)
$$

**含义**: 以当前状态、动作和隐式条件为条件，稀疏预测未来视觉帧

**符号说明**:
- $\Delta$: 预测帧间的时间步长
- $K = \lfloor p / \Delta \rfloor$: 预测的未来帧数

### 公式 4-5: [[Flow Matching|流匹配插值]]

$$
x^{(s)} = (1-s)\varepsilon + sx
$$

$$
\dot{x}^{(s)} = x - \varepsilon
$$

**含义**: 构造流匹配训练中的插值噪声变量和目标速度场

**符号说明**:
- $s \sim \mathcal{U}(0, 1)$: 流时间参数
- $\varepsilon \sim \mathcal{N}(0, I)$: 噪声
- $x$: 目标变量（动作或视频隐变量）
- $\dot{x}^{(s)}$: 目标速度

### 公式 5: [[Flow Matching|视频流匹配损失]]

$$
\mathcal{L}_{video} = \mathbb{E}_{s, \varepsilon}\Big[\big\|g_\Theta(z_f^{(s)}, s \mid T_s, T_o, T_a, T_l) - \dot{z}_f^{(s)}\big\|^2\Big]
$$

**含义**: 训练模型预测未来观测隐变量的速度场，以历史和动作为条件

**符号说明**:
- $z_f$: 未来观测的 [[VAE]] 隐变量
- $T_l$: 语言指令 token

### 公式 6: [[Flow Matching|动作流匹配损失]]

$$
\mathcal{L}_{action} = \mathbb{E}_{s, \varepsilon}\Big[\big\|g_\Theta(a^{(s)}, s \mid T_s, T_o, T_l) - \dot{a}^{(s)}\big\|^2\Big]
$$

**含义**: 优化以观测和语言为条件的动作预测

### 公式 7: [[Multi-Task Loss|联合训练损失]]

$$
\mathcal{L}_{all} = \lambda_{video} \mathcal{L}_{video} + \lambda_{action} \mathcal{L}_{action}
$$

**含义**: 后训练阶段通过加权组合平衡视频与动作目标

**符号说明**:
- $\lambda_{video} = 1$: 视频损失权重
- $\lambda_{action} = 5$: 动作损失权重（强调动作预测）

### 公式 8: [[Flow Matching|推理时动作 ODE 积分]]

$$
\frac{da^{(s)}}{ds} = g_\Theta(a^{(s)}, s \mid w_t), \quad s \in [0, 1]
$$

其中 $w_t = (T_l, T_s, T_o)$ 为推理时的条件上下文。

**含义**: 推理时从纯噪声 $a^{(0)} \sim \mathcal{N}(0, I)$ 积分到 $a^{(1)}$，得到动作块

---

## 关键图表

### Figure 1: Comparison / 基线对比

![Figure 1](https://arxiv.org/html/2603.17240v2/x1.png)

**说明**: GigaWorld-Policy 与基线在推理频率和真实世界任务成功率上的对比。GigaWorld-Policy 在 A100 上仅需 360ms 推理延迟，成功率达 0.83，远优于 Motus（3231ms, 0.76）和 π₀.₅（225ms, 0.69）。

### Figure 2: Method Taxonomy / 方法分类

![Figure 2](https://arxiv.org/html/2603.17240v2/x2.png)

**说明**: 四类方法范式对比：(a) VLM-based VLA 辅以未来监督；(b) 双向注意力联合动作-视频预测（推理时必须生成视频）；(c) 两阶段先生成视频再用 [[IDM|逆动力学模型]] 解码动作；(d) GigaWorld-Policy 的以动作为中心设计——训练时利用视频监督，推理时视频可选。

### Figure 3: Overview / 系统概览

![Figure 3](https://arxiv.org/html/2603.17240v2/x3.png)

**说明**: GigaWorld-Policy 整体架构。预训练阶段从大规模视频学习动作相关表征；后训练阶段联合执行 [[Action Chunking|动作块预测]] 和未来视频预测作为辅助监督；推理时未来视频分支可选，实现更快推理。

### Figure 4: Attention Mask / 注意力掩码

![Figure 4](https://arxiv.org/html/2603.17240v2/x4.png)

**说明**: GigaWorld-Policy 的块级 [[Causal Attention|因果注意力掩码]]。动作 token $T_a$ 仅注意状态 $T_s$ 和当前观测 $T_o$；未来视频 token $T_f$ 额外注意动作 token。这保证了推理时动作生成不依赖未来视频。

### Figure 5: Real-World QR Code Scanning / 真实世界扫码任务

![Figure 5](https://arxiv.org/html/2603.17240v2/x5.png)

**说明**: GigaWorld-Policy 在 [[AgileX PiPER]] 双臂上执行二维码扫描任务的实际部署。

### Figure 6: Real-World Trash Sweeping / 真实世界扫垃圾任务

![Figure 6](https://arxiv.org/html/2603.17240v2/x6.png)

**说明**: GigaWorld-Policy 在 [[AgileX PiPER]] 双臂上执行扫垃圾任务的实际部署。

### Figure 7: Data Efficiency / 数据效率

![Figure 7](https://arxiv.org/html/2603.17240v2/x7.png)

**说明**: 成功率随训练数据比例的变化。GigaWorld-Policy 仅用 10% 的数据即可匹配 π₀.₅ 的性能，展示了优异的数据效率。

### Figure 8: Pre-training Data Scaling / 预训练数据扩展

![Figure 8](https://arxiv.org/html/2603.17240v2/x8.png)

**说明**: 增加具身数据预训练比例持续提升真实世界成功率，验证了大规模具身数据预训练的价值。

### Figure 9: Qualitative Comparison / 定性对比

![Figure 9](https://arxiv.org/html/2603.17240v2/x9.png)

**说明**: 自注意力基线与本文因果掩码方法的视频生成质量对比。红色框中，因果掩码方法更准确地预测了物体状态变化（PSNR 28.41 vs. 27.87）。

### Figure 10: Real-World Additional Tasks / 更多真实世界任务

![Figure 10](https://arxiv.org/html/2603.17240v2/x10.png)

**说明**: GigaWorld-Policy 在 [[AgileX PiPER]] 双臂上执行碗堆叠和桌面清洁任务的实际部署。

### Table 1: 具身数据预训练数据集

| 数据集 | 时长 | 数据集 | 时长 | 数据集 | 时长 |
|--------|------|--------|------|--------|------|
| EgoDex | 800h | Agibot | 2,500h | EGO4D | 3,500h |
| Open X-Embodiment | 3,500h | RoboMind | 300h | RDT | 25h |
| Something-Something V2 | 200h | DROID | 350h | ATARA | 10h |

**说明**: 具身数据预训练阶段使用约 10,000 小时数据，涵盖真实机器人视频、第一人称人类演示和交互视频数据集。

### Table 2: RoboTwin 2.0 仿真评估（50 任务）

| 任务 | π₀.₅ Clean | π₀.₅ Rand. | X-VLA Clean | X-VLA Rand. | Motus Clean | Motus Rand. | **Ours Clean** | **Ours Rand.** |
|------|-----------|-----------|------------|------------|------------|------------|---------------|---------------|
| Adjust Bottle | 0.79 | 0.83 | 1.00 | 0.99 | 0.89 | 0.93 | **1.00** | **1.00** |
| Place A2b Left | 0.62 | 0.60 | 0.48 | 0.49 | 0.88 | 0.79 | **0.94** | **0.88** |
| Place Bread Skillet | 0.38 | 0.46 | 0.77 | 0.67 | 0.86 | 0.83 | **0.94** | **0.90** |
| Place Cans Plasticbox | 0.40 | 0.47 | 0.97 | 0.98 | 0.98 | 0.94 | **1.00** | **1.00** |
| Place Fan | 0.25 | 0.36 | 0.80 | 0.75 | 0.91 | 0.87 | **0.92** | **0.94** |
| Place Mouse Pad | 0.21 | 0.26 | 0.70 | 0.70 | 0.66 | 0.68 | **0.88** | **0.90** |
| Place Object Basket | 0.43 | 0.36 | 0.44 | 0.39 | 0.81 | 0.87 | **0.90** | **0.92** |
| Place Object Stand | 0.74 | 0.65 | 0.86 | 0.88 | 0.98 | 0.97 | **1.00** | **0.98** |
| Rotate Qrcode | 0.47 | 0.56 | 0.34 | 0.33 | 0.89 | 0.73 | **0.90** | **0.84** |
| Shake Bottle | 0.91 | 1.00 | 0.99 | 1.00 | 1.00 | 0.97 | **1.00** | **1.00** |
| Stamp Seal | 0.36 | 0.23 | 0.76 | 0.82 | 0.93 | 0.92 | **0.96** | **0.98** |
| Dump Bin Bigbin | 0.30 | 0.42 | 0.79 | 0.77 | 0.95 | 0.91 | **0.92** | **1.00** |
| Handover Block | 0.18 | 0.19 | 0.73 | 0.37 | 0.86 | 0.73 | **0.80** | **0.80** |
| Handover Mic | 0.28 | 0.18 | 0.00 | 0.00 | 0.78 | 0.63 | **0.72** | **0.72** |
| Move Stapler Pad | 0.16 | 0.18 | 0.78 | 0.73 | 0.83 | 0.85 | **0.92** | **0.82** |
| Open Laptop | 0.19 | 0.35 | 0.93 | 1.00 | 0.95 | 0.91 | **0.96** | **0.98** |
| Place A2b Right | 0.62 | 0.57 | 0.36 | 0.36 | 0.91 | 0.87 | **0.90** | **0.92** |
| Place Burger Fries | 0.66 | 0.70 | 0.94 | 0.94 | 0.98 | 0.98 | **0.98** | **0.96** |
| Place Container Plate | 0.71 | 0.78 | 0.97 | 0.95 | 0.98 | 0.99 | **0.98** | **0.96** |
| Press Stapler | 0.80 | 0.70 | 0.92 | 0.98 | 0.93 | 0.98 | **0.96** | **0.96** |
| Put Object Cabinet | 0.24 | 0.15 | 0.46 | 0.48 | 0.88 | 0.71 | **0.74** | **0.74** |
| Place Dual Shoes | 0.12 | 0.07 | 0.79 | 0.88 | 0.93 | 0.87 | **0.96** | **0.84** |
| Stack Blocks Two | 0.48 | 0.56 | 0.92 | 0.87 | 1.00 | 0.98 | **1.00** | **0.94** |
| Turn Switch | 0.05 | 0.06 | 0.40 | 0.61 | 0.84 | 0.78 | **0.82** | **0.84** |
| **Average** | 0.43 | 0.44 | 0.73 | 0.73 | 0.89 | 0.87 | **0.87** | **0.85** |

**说明**: 在 RoboTwin 2.0 仿真的 24 个代表性任务上（Clean = 干净背景，Rand. = 随机化背景），GigaWorld-Policy 平均成功率 0.86 与 Motus (0.88) 相当，远超 π₀.₅ (0.44) 和 X-VLA (0.73)。

### Table 3: 推理延迟与成功率

| 方法 | 推理时间 (ms) | 仿真成功率 | 真实世界成功率 |
|------|-------------|-----------|--------------|
| π₀.₅ | 225 | 0.48 | 0.69 |
| GigaBrain-0 | 452 | — | 0.68 |
| Motus | 3231 | 0.88 | 0.76 |
| Cosmos-Policy | 1413 | — | 0.58 |
| **Ours** | **360** | **0.86** | **0.83** |

**说明**: GigaWorld-Policy 推理延迟仅 360ms，比 Motus 快 9 倍，同时真实世界成功率最高（0.83）。

### Table 4: 真实世界任务成功率

| 方法 | 桌面清洁 | 扫码 | 扫垃圾 | 碗堆叠 | 平均 |
|------|---------|------|--------|--------|------|
| π₀.₅ | 0.75 | 0.55 | 0.65 | 0.80 | 0.69 |
| GigaBrain-0 | 0.70 | 0.65 | 0.60 | 0.75 | 0.68 |
| Motus | 0.80 | 0.75 | 0.70 | 0.80 | 0.76 |
| Cosmos-Policy | 0.65 | 0.50 | 0.45 | 0.70 | 0.58 |
| **Ours** | **0.90** | **0.75** | **0.75** | **0.90** | **0.83** |

**说明**: 在 [[AgileX PiPER]] 双臂上的四项真实世界任务中，GigaWorld-Policy 平均成功率 0.83，超越 Motus 7 个百分点。桌面清洁和碗堆叠任务尤为突出（0.90）。

### Table 5: 采样间隔 $\Delta$ 消融

| $\Delta$ | 0 | 4 | 8 | 12 | 24 | 48 |
|----------|---|---|---|----|----|-----|
| 成功率 ↑ | 0.60 | 0.76 | 0.78 | **0.83** | 0.80 | 0.76 |

**说明**: $\Delta = 0$ 表示不使用视频监督，成功率仅 0.60。$\Delta = 12$ 为最优间隔，过密（$\Delta=4$）导致冗余监督，过稀（$\Delta=48$）丢失关键演化信息。

### Table 6: 因果注意力掩码消融

| 方法 | 成功率 ↑ | PSNR ↑ | SSIM ↑ |
|------|---------|--------|--------|
| Self-Attn | 0.81 | 27.87 | 0.892 |
| **Ours (Causal)** | **0.83** | **28.41** | **0.901** |

**说明**: 因果注意力掩码相比全自注意力不仅提升了动作成功率（+2%），还改善了视频生成质量（PSNR +0.54, SSIM +0.009）。

### Table 7: 预训练配置消融

| 预训练初始化 | 具身数据预训练 | 成功率 ↑ |
|------------|-------------|---------|
| — | — | 0.45 |
| ✓ | — | 0.57 |
| — | ✓ | 0.73 |
| ✓ | ✓ | **0.83** |

**说明**: 网络视频预训练和具身数据预训练提供互补收益。两者结合从 0.45 提升到 0.83，具身数据预训练贡献最大（+0.28）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Web 视频 | 大规模 | 多样自然视频 | Stage 1 基础预训练 |
| 具身数据（10 种来源） | ~10,000h | 机器人+人类演示 | Stage 2 具身预训练 |
| RoboTwin 2.0 | 50 任务 | 仿真双臂操作 | 仿真评估 |
| AgileX PiPER 真实任务 | 4 任务 | 双臂真实世界操作 | 真实世界评估 |

### 实现细节

- **Backbone**: 5B 参数 [[Diffusion Transformer]]（视频生成模型预训练）
- **优化器**: [[AdamW]]，$\beta_1=0.85$, $\beta_2=0.9$
- **学习率**: 余弦衰减，$1 \times 10^{-4} \to 1 \times 10^{-6}$
- **Batch Size**: 256
- **训练消耗**: 6000 GPU hours
- **损失权重**: $\lambda_{action} = 5$, $\lambda_{video} = 1$
- **硬件**: A100 GPU

### 推理流程

1. 构建条件上下文 $w_t = (T_l, T_s, T_o)$
2. 初始化动作噪声 $a^{(0)} \sim \mathcal{N}(0, I)$
3. 对流 ODE 从 $s=0$ 积分到 $s=1$
4. 解码 $a^{(1)}$ 为连续动作块
5. 执行动作，下一控制步重复

### 可视化结果

- 因果掩码方法在视频生成中更准确地预测物体状态变化（Figure 9）
- 仅需 10% 训练数据即可匹配 π₀.₅ 的完整数据性能（Figure 7）
- 具身预训练数据量与性能呈正相关（Figure 8）

---

## 批判性思考

### 优点

1. **优雅的解耦设计**: 因果注意力掩码在训练时利用视频监督、推理时跳过视频生成，是简洁有效的解决方案
2. **全面的实验验证**: 50 任务仿真 + 4 任务真实世界 + 多维度消融实验，说服力强
3. **实用性突出**: 360ms 的推理延迟和 0.83 的真实成功率使其具备真正的部署潜力
4. **数据效率高**: 10% 数据即可匹配基线，降低了数据收集成本

### 局限性

1. **真实世界任务有限**: 仅 4 项双臂任务，缺乏灵巧手、移动操作等更复杂场景的验证
2. **仿真-真实差距**: 仿真中 Motus (0.88) 略优于 GigaWorld-Policy (0.86)，真实世界反转，说明仿真不能完全预测真实表现
3. **5B 参数量较大**: 虽然推理快，但模型体积对边缘部署仍有挑战
4. **视频生成质量分析不足**: 仅在消融中简略比较了 PSNR/SSIM，未深入分析视频辅助监督的具体贡献机制

### 潜在改进方向

1. **轻量化**: 探索模型蒸馏或剪枝以适应边缘部署
2. **更多任务类型**: 扩展到灵巧手操作、移动操作、多机器人协作等场景
3. **自适应 $\Delta$**: 根据任务复杂度动态调整未来帧预测间隔
4. **闭环视频生成**: 研究何时启用可选的视频生成能进一步提升性能

### 可复现性评估

- [x] 代码开源（[GitHub](https://github.com/open-gigaai/giga-world-policy)）
- [ ] 预训练模型（[HuggingFace](https://huggingface.co/open-gigaai)，待确认）
- [x] 训练细节完整（优化器、学习率、batch size、GPU hours 均提供）
- [ ] 数据集可获取（部分开源数据集可获取，但 Agibot 等可能受限）

---

## 关联笔记

### 基于
- [[Diffusion Transformer]]: 5B 参数视频生成骨干网络
- [[Flow Matching]]: 连续时间流匹配训练框架
- [[Action Chunking]]: 动作块预测范式

### 对比
- [[Motus]]: 联合动作-视频预测但推理需生成视频（3231ms vs. 360ms）
- [[Pi05|π₀.₅]]: 纯动作 VLA，推理快但成功率低（0.69 vs. 0.83）
- [[X-VLA]]: VLM-based VLA 基线
- [[Cosmos-Policy]]: 两阶段世界模型方法

### 方法相关
- [[WAM]]: 世界-动作模型核心概念
- [[VLA]]: 视觉-语言-动作模型
- [[Causal Attention]]: 核心因果注意力掩码设计
- [[VAE]]: 视觉 token 编码器
- [[Cross-Attention]]: 语言条件注入方式

### 硬件/数据相关
- [[AgileX PiPER]]: 真实世界实验平台（双臂机器人）
- [[Open X-Embodiment]]: 大规模机器人数据集
- [[EGO4D]]: 第一人称人类活动视频数据集

---

## 速查卡片

> [!summary] GigaWorld-Policy
> - **核心**: 以动作为中心的世界-动作模型，推理时可选地跳过视频生成
> - **方法**: 因果注意力掩码 + 三阶段预训练 + 流匹配
> - **结果**: 比 Motus 快 9 倍（360ms vs. 3231ms），真实世界成功率 0.83（+7%）
> - **代码**: [GitHub](https://github.com/open-gigaai/giga-world-policy)

---

*笔记创建时间: 2026-04-24*
