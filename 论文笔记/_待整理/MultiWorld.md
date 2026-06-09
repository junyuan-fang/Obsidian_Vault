---
title: "MultiWorld: Scalable Multi-Agent Multi-View Video World Models"
method_name: "MultiWorld"
authors: [Haoyu Wu, Jiwen Yu, Yingtian Zou, Xihui Liu]
year: 2026
venue: arXiv
tags: [world-model, multi-agent, multi-view, video-generation, diffusion-model, robot-simulation, flow-matching]
zotero_collection: _待整理
image_source: online
arxiv_html: https://arxiv.org/html/2604.18564v2
created: 2026-04-24
---

# 论文笔记：MultiWorld: Scalable Multi-Agent Multi-View Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Hong Kong, Sreal AI |
| 日期 | April 2026 |
| 项目主页 | https://multi-world.github.io/ |
| 对比基线 | COMBO, Concat-View |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18564) / [Project](https://multi-world.github.io/) |

---

## 一句话总结

> 提出统一框架 MultiWorld，通过多智能体条件模块和全局状态编码器实现可扩展的多智能体多视角视频世界模型。

---

## 核心贡献

1. **Multi-Agent Condition Module (MACM)**: 通过 Agent Identity Embedding 和 Adaptive Action Weighting 实现多智能体动作的精确可控性
2. **Global State Encoder**: 利用预训练的 [[VGGT]] 提取 3D 感知全局状态，通过 [[Cross-Attention]] 注入生成过程以保证多视角一致性
3. **可扩展框架设计**: 支持可变数量的智能体和视角，通过相对位置编码和全局状态压缩实现灵活扩展

---

## 问题背景

### 要解决的问题

现有的视频世界模型主要面向单智能体、单视角场景，无法同时处理多个智能体的动作控制和多视角之间的几何一致性。多智能体多视角世界建模面临三大核心挑战：(1) 多智能体可控性 (2) 多视角一致性 (3) 框架可扩展性。

### 现有方法的局限

- **单智能体假设**: 大多数视频世界模型（如 GameGen、Oasis）仅支持单一智能体控制，无法建模多智能体交互
- **固定视角数量**: 现有多视角方法假设固定相机数量，缺乏灵活性
- **简单拼接策略**: 将多视角图像简单拼接为宽图（Concat-View）会导致视角间一致性差
- **动作偏差问题**: 已有动作条件化世界模型经常在静态动作输入时仍产生运动（action bias）

### 本文的动机

作者观察到多智能体场景中，不同智能体的动作具有不对称性（某些时刻只有部分智能体在活跃），且多视角之间共享相同的 3D 场景状态。因此提出：(1) 用类似 [[RoPE]] 的旋转位置编码区分不同智能体身份 (2) 用自适应权重放大活跃智能体的影响 (3) 用预训练 3D 视觉模型压缩全局状态来同步各视角。

---

## 方法详解

### 模型架构

MultiWorld 基于 [[Wan2.1|Wan2.2-5B]] 视频生成模型，采用 [[Flow Matching]] + [[Transformer]] 架构：

- **输入**: 多视角初始帧 $\{o_1, o_2, \dots, o_V\}$ + 每帧每智能体动作 $\{a_1, a_2, \dots, a_N\}$
- **Backbone**: Wan2.2-5B [[DiT]] 模型（预训练文本到视频生成器）
- **核心模块 1**: Multi-Agent Condition Module（MACM）用于动作条件化
- **核心模块 2**: Global State Encoder（GSE）用于多视角一致性
- **输出**: 多视角一致的动作可控视频帧序列
- **训练分辨率**: 320x320（游戏场景 320x640），81 帧

### 核心模块

#### 模块 1: Multi-Agent Condition Module (MACM)

**设计动机**: 多智能体动作条件化需要解决两个子问题——如何区分不同智能体的身份（打破对称性），以及如何动态调整不同智能体的影响力。

**Agent Identity Embedding (AIE)**:

借鉴 [[RoPE|Rotary Position Embedding]] 的思路，使用旋转矩阵对每个智能体的动作向量进行身份编码。不同于传统的加性位置编码，旋转编码能在注意力计算中自然地表达智能体间的相对关系。

具体实现：
- 将智能体索引 $i$ 映射为旋转角度 $\theta_j = b^{-2j/D}$，其中 $b$ 是基频超参数
- 对动作向量的每对维度施加 2D 旋转变换
- 基频 $b=20$ 优于标准的 $b=10000$，因为智能体数量远小于 token 序列长度

**Adaptive Action Weighting (AAW)**:

通过学习一个权重网络动态调节每个智能体动作的影响力。直觉是：当某智能体静止时，其动作信号应被弱化；当智能体活跃时，信号应被放大。这有助于减轻 action bias 问题。

#### 模块 2: Global State Encoder (GSE)

**设计动机**: 利用 [[VGGT]]（Visual Geometry Grounded Transformer）提取多视角观测的 3D 感知特征，为生成过程提供统一的全局场景状态。

**具体实现**:
- 冻结预训练 [[VGGT]] 模型，输入所有视角的条件帧
- 提取 3D 感知的全局状态特征 $g$
- 通过可学习的 [[Cross-Attention]] 层将 $g$ 注入 [[DiT]] 的每个 Transformer block
- 全局状态压缩了多视角几何信息，使各视角生成时共享相同的 3D 理解

#### 可扩展设计

- **智能体扩展**: AIE 使用相对位置编码，支持任意数量智能体
- **视角扩展**: 各视角独立生成但共享全局状态，通过并行生成实现高效推理
- **长时序生成**: 自回归分块策略（autoregressive chunking），用前一段末尾帧作下一段条件

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配前向过程]]

$$
\mathbf{x}_c^t = (1 - t)\mathbf{x}_c + t\boldsymbol{\epsilon}
$$

**含义**: 在时间步 $t$ 对干净观测 $\mathbf{x}_c$ 添加噪声，构建从数据分布到噪声分布的插值路径。

**符号说明**:
- $\mathbf{x}_c$: 视角 $c$ 的干净观测帧
- $t \in [0, 1]$: 流匹配时间步
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: 标准高斯噪声
- $\mathbf{x}_c^t$: 时间步 $t$ 的噪声观测

### 公式 2: [[Flow Matching|目标速度场]]

$$
\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{x}_c
$$

**含义**: 定义从数据到噪声的线性速度场，作为速度网络的学习目标。

### 公式 3: [[RoPE|Agent Identity Embedding 旋转变换]]

$$
\text{AIE}(\mathbf{a}_i, i) = \mathbf{R}_{\Theta, i} \mathbf{a}_i
$$

**含义**: 通过旋转矩阵将智能体索引编码到动作向量中，打破多智能体间的对称性。

**符号说明**:
- $\mathbf{a}_i$: 第 $i$ 个智能体的动作向量
- $i$: 智能体索引
- $\mathbf{R}_{\Theta, i}$: 由参数 $\Theta$ 和索引 $i$ 决定的旋转矩阵

### 公式 4: [[RoPE|旋转矩阵定义]]

$$
\begin{bmatrix} a_{\text{out}}^{(2j)} \\ a_{\text{out}}^{(2j+1)} \end{bmatrix} = \begin{bmatrix} \cos(i\theta_j) & -\sin(i\theta_j) \\ \sin(i\theta_j) & \cos(i\theta_j) \end{bmatrix} \begin{bmatrix} a_{\text{in}}^{(2j)} \\ a_{\text{in}}^{(2j+1)} \end{bmatrix}
$$

**含义**: 对动作向量的每对维度 $(2j, 2j+1)$ 施加依赖智能体索引 $i$ 的 2D 旋转。

**符号说明**:
- $\theta_j = b^{-2j/D}$: 第 $j$ 对维度的旋转频率
- $b = 20$: 基频超参数（远小于标准 RoPE 的 10000）
- $D$: 动作嵌入维度

### 公式 5: [[Attention|相对位置注意力性质]]

$$
(\mathbf{R}_m \mathbf{a}_m)^T (\mathbf{R}_n \mathbf{a}_n) = \mathbf{a}_m^T \mathbf{R}_{n-m} \mathbf{a}_n
$$

**含义**: 旋转编码在点积注意力中自然退化为相对位置编码，使模型关注智能体间的相对关系而非绝对索引。

### 公式 6: [[Flow Matching|流匹配训练损失]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \boldsymbol{\epsilon}} \Big[ \lVert v_\theta(\mathbf{x}^t, t, \mathbf{a}) - \mathbf{u} \rVert_2^2 \Big]
$$

**含义**: 最小化速度网络预测与目标速度场之间的均方误差。

**符号说明**:
- $v_\theta$: 参数为 $\theta$ 的速度网络
- $\mathbf{a}$: 所有智能体的动作条件
- $\lVert \cdot \rVert_2^2$: L2 范数平方

### 公式 7: [[ODE|采样 ODE]]

$$
d\mathbf{x} = v_\theta(\mathbf{x}^t, t, \mathbf{a}) \cdot dt
$$

**含义**: 推理时通过求解常微分方程从噪声采样生成视频帧。

### 公式 8: [[Reprojection Error|重投影误差 RPE]]

$$
\text{RPE} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \lVert p^*_{ij} - \Pi(P_{ij}) \rVert_2
$$

**含义**: 衡量多视角间几何一致性的指标，通过匹配点的重投影误差来评估视角间的 3D 一致性。

**符号说明**:
- $\mathcal{V}$: 视角对集合
- $p^*_{ij}$: 视角 $j$ 中匹配点的真实位置
- $\Pi(P_{ij})$: 3D 点 $P_{ij}$ 到视角 $j$ 的投影
- $\lVert \cdot \rVert_2$: L2 范数

---

## 关键图表

### Figure 1: Teaser / 多智能体多视角生成展示

![Figure 1](https://arxiv.org/html/2604.18564v2/x1.png)

**说明**: MultiWorld 的核心能力展示。给定初始视角和每帧动作，模型在多人游戏（It Takes Two）和多机器人操作场景中生成动作可控、多视角一致的视频。上排展示双人游戏中两个玩家的独立控制，下排展示多机器人协作操作。

### Figure 2: Pipeline / 模型流程

![Figure 2](https://arxiv.org/html/2604.18564v2/x2.png)

**说明**: MultiWorld 的完整流程。左侧为 Multi-Agent Condition Module，包含 Agent Identity Embedding（通过旋转矩阵编码智能体身份）和 Adaptive Action Weighting（动态调节活跃智能体权重）。右侧为 Global State Encoder，利用冻结的 [[VGGT]] 提取 3D 感知全局状态并通过 [[Cross-Attention]] 注入 [[DiT]] 生成过程。

### Figure 3: Qualitative Comparison / 定性对比

![Figure 3](https://arxiv.org/html/2604.18564v2/x3.png)

**说明**: 多人游戏场景的定性对比。与 Standard baseline、Concat-View、COMBO 相比，MultiWorld 在动作跟随精度和多视角一致性上均表现更优。其他方法存在视角间物体位置不一致或动作响应不准确的问题。

### Figure 4: Multi-Robot Failure Simulation / 多机器人失败轨迹仿真

![Figure 4](https://arxiv.org/html/2604.18564v2/x4.png)

**说明**: MultiWorld 模拟真实的协作失败场景，如多机器人协作操作中的碰撞。这体现了模型对物理交互的理解能力，对于机器人策略学习中的安全分析具有重要价值。

### Figure 5: Long-Horizon Generation / 长时序生成

![Figure 5](https://arxiv.org/html/2604.18564v2/x5.png)

**说明**: 多机器人操作任务的长时序视频生成。模型通过自回归分块策略，让三个机器人按序堆叠彩色方块，在较长时间跨度内保持连贯性和动作准确性。

### Figure 6: Action Controllability / 动作可控性

![Figure 6](https://arxiv.org/html/2604.18564v2/x6.png)

**说明**: MultiWorld 在给定静态动作时忠实地生成静态视频，而其他动作条件化世界模型常存在 action bias（即使输入静态动作也会产生虚假运动）。这得益于 Adaptive Action Weighting 模块。

### Figure 7: Multi-Agent Environment Interactions / 多智能体环境交互

![Figure 7](https://arxiv.org/html/2604.18564v2/x7.png)

**说明**: MultiWorld 准确模拟涉及多智能体的复杂环境交互。例如一个智能体推动而另一个拉动大型板材的协作场景，展现了模型对物理因果关系的建模能力。

### Figure 8: Physical Consistency / 物理一致性

![Figure 8](https://arxiv.org/html/2604.18564v2/x8.png)

**说明**: MultiWorld 生成的多视角视频在物理效果上保持一致。上方案例中，物体投射的阴影在两个相对视角中保持一致，说明模型通过 Global State Encoder 学习到了共享的 3D 场景理解。

### Figure 9: Failure Cases / 失败案例分析

![Figure 9](https://arxiv.org/html/2604.18564v2/x9.png)

**说明**: 当智能体在视野中占据较小区域时（远距离物体），由于空间分辨率限制，生成的智能体往往显得模糊。这是当前模型的主要局限性之一。

### Figure 10: Multi-Robot Success Trajectory / 多机器人成功轨迹

![Figure 10](https://arxiv.org/html/2604.18564v2/x10.png)

**说明**: MultiWorld 生成物理上合理的多机器人协作行为，机器人成功协调完成操作任务，展示了模型作为世界模拟器的潜力。

### Table 1: 多视角条件化策略对比

**Multi-Player Video Game:**

| Method | FVD$\downarrow$ | LPIPS$\downarrow$ | SSIM$\uparrow$ | PSNR$\uparrow$ | Action$\uparrow$ | RPE$\downarrow$ |
|--------|------|--------|-------|-------|---------|------|
| Standard | 245 | 0.36 | 0.50 | 17.48 | 88.4 | 0.75 |
| Concat-View | 215 | 0.36 | 0.49 | 17.54 | 89.1 | 0.74 |
| COMBO | 207 | 0.34 | 0.51 | 17.82 | 89.3 | 0.72 |
| **MultiWorld** | **179** | 0.35 | 0.51 | 17.72 | **89.8** | **0.67** |

**Multi-Robot Manipulation:**

| Method | FVD$\downarrow$ | LPIPS$\downarrow$ | SSIM$\uparrow$ | PSNR$\uparrow$ | Action$\uparrow$ | RPE$\downarrow$ |
|--------|------|--------|-------|-------|---------|------|
| Standard | 100 | 0.07 | 0.90 | 26.39 | 88.2 | 1.60 |
| Concat-View* | 106 | 0.06 | 0.90 | 27.44 | 92.0 | 0.82 |
| COMBO | 99 | 0.08 | 0.90 | 26.49 | 88.5 | 1.54 |
| **MultiWorld** | **96** | 0.07 | 0.90 | 26.60 | 88.7 | **1.52** |

**说明**: MultiWorld 在两个场景中均取得最优 FVD 和 RPE（多视角一致性）。游戏场景中 FVD 从 207 降至 179（13.5% 提升），RPE 从 0.72 降至 0.67（6.9% 提升）。

### Table 2: 主要组件消融实验

| Config | FVD$\downarrow$ | LPIPS$\downarrow$ | SSIM$\uparrow$ | PSNR$\uparrow$ | Action$\uparrow$ | RPE$\downarrow$ |
|--------|------|--------|-------|-------|---------|------|
| Standard | 245 | 0.36 | 0.50 | 17.48 | 88.4 | 0.75 |
| + MACM | 228 | 0.36 | 0.51 | 17.56 | 89.7 | 0.76 |
| + MACM + GSE (Full) | **179** | **0.35** | 0.51 | **17.72** | **89.8** | **0.67** |

**关键发现**: MACM 主要提升动作可控性（Action 88.4 -> 89.7），GSE 主要提升多视角一致性（RPE 0.76 -> 0.67），两者互补。

### Table 3: Agent Identity Embedding 基频消融

| Config | FVD$\downarrow$ | PSNR$\uparrow$ | Action$\uparrow$ |
|--------|------|-------|---------|
| base=10000 | 234 | 17.53 | 89.2 |
| base=20 | **228** | **17.56** | **89.7** |

**关键发现**: 较小的基频（$b=20$）优于标准 RoPE 的 $b=10000$，因为智能体数量（2-4）远小于 NLP 中的 token 序列长度。

### Table 4: Adaptive Action Weighting 消融

| Config | FVD$\downarrow$ | PSNR$\uparrow$ | Action$\uparrow$ |
|--------|------|-------|---------|
| w/o AAW | 245 | 17.48 | 88.4 |
| w/ AAW | **236** | **17.52** | **88.6** |

**关键发现**: AAW 通过动态调节活跃智能体权重，减轻了 action bias，改善了整体视频质量。

### Table 5: Global State Encoder 消融

| Global State Encoder | FVD$\downarrow$ | LPIPS$\downarrow$ | SSIM$\uparrow$ | PSNR$\uparrow$ | RPE$\downarrow$ |
|----------------------|------|--------|-------|-------|------|
| w/o Global State | 228 | 0.36 | 0.51 | 17.56 | 0.75 |
| Wan VAE | 256 | 0.36 | 0.50 | 17.38 | 0.71 |
| DINOv2 | 232 | 0.36 | 0.50 | 17.48 | 0.72 |
| **VGGT (Ours)** | **179** | **0.35** | **0.51** | **17.72** | **0.67** |

**关键发现**: [[VGGT]] 的 3D 感知特征显著优于 2D 特征提取器（[[DINO|DINOv2]]）和通用 VAE 编码器，证明了 3D 几何理解对多视角一致性的关键作用。VGGT 将 RPE 从 0.75 降至 0.67（10.7% 提升）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| It Takes Two | 100 小时（21M 帧） | 双人合作游戏，2560x1440，双视角 | 训练+测试 |
| RoboFactory | 4 任务 x 3000 episodes | 多机器人操作，多视角，含成功和失败轨迹 | 训练+测试 |

**It Takes Two 数据集**: 从 500 小时原始录制中筛选 100 小时高质量片段，包含 21M 帧，原始分辨率 2560x1440，训练时降采样至 320x640。每帧记录双玩家的独立游戏手柄输入。

**RoboFactory 数据集**: 四个任务——击球（striking）、2 机器人堆叠、3 机器人堆叠、4 机器人传递。每任务 3000 episodes（1000 成功 + 2000 控制失败），训练分辨率 256x320。

### 实现细节

- **Backbone**: [[Wan2.1|Wan2.2-5B]]（预训练文本到视频 [[DiT]]）
- **优化器**: AdamW
- **训练帧数**: 81 帧
- **训练分辨率**: 320x320（机器人），320x640（游戏）
- **训练迭代**: 40k iterations
- **训练时长**: 4 天
- **硬件**: 8x NVIDIA A800 GPU
- **Global State Encoder**: 冻结的 [[VGGT]]，通过可学习 [[Cross-Attention]] 注入
- **评估指标**: [[FVD]]、[[LPIPS]]、[[SSIM]]、[[PSNR]]、RPE（重投影误差）、Inverse Dynamics Model accuracy

### 可视化结果

- 多人游戏场景中，MultiWorld 准确响应两个玩家的独立动作输入，视角间物体位置一致
- 多机器人场景中，模型成功模拟 2-4 个机器人的协作操作，包括成功和失败轨迹
- 长时序自回归生成保持了时间连贯性和动作准确性
- 物理效果（如阴影）在不同视角间保持一致

---

## 批判性思考

### 优点
1. **统一框架**: 同时解决多智能体可控性和多视角一致性，而非分别处理
2. **巧妙的 RoPE 迁移**: 将 NLP 中的旋转位置编码创新地应用于智能体身份编码，理论上支持任意数量智能体
3. **冻结 VGGT 策略**: 利用预训练 3D 视觉模型提供几何先验，避免从零学习 3D 一致性
4. **数据集贡献**: 构建了两个高质量的多智能体多视角数据集，填补了社区空白
5. **实际应用价值**: 多机器人失败轨迹仿真对安全分析和策略学习有直接意义

### 局限性
1. **分辨率受限**: 训练分辨率仅为 320x320/640，远低于实际应用需求
2. **计算成本**: 基于 5B 参数的 DiT 模型，推理速度难以满足实时需求
3. **远距离物体模糊**: 当智能体在视野中占比小时生成质量下降
4. **训练规模有限**: 仅在两个特定场景上验证，泛化能力未经充分测试
5. **缺乏真实世界验证**: 仅在仿真/游戏环境中评估，未在真实多机器人系统上测试

### 潜在改进方向
1. 引入记忆机制支持超长时序生成
2. 结合 [[3DGS]] 实现更精确的几何一致性约束
3. 探索实时推理优化（如模型蒸馏、量化）
4. 扩展到真实世界多机器人场景

### 可复现性评估
- [x] 代码开源（GitHub）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整
- [x] 数据集可获取（Hugging Face）

---

## 关联笔记

### 基于
- [[Wan2.1]]: 基础视频生成模型（Wan2.2-5B）
- [[VGGT]]: Global State Encoder 的 3D 视觉骨干

### 对比
- [[COMBO]]: 多视角视频生成基线方法
- [[GameGen]]: 单智能体游戏世界模型

### 方法相关
- [[Flow Matching]]: 核心生成范式
- [[RoPE]]: Agent Identity Embedding 的理论基础
- [[Cross-Attention]]: Global State 注入机制
- [[DiT]]: Transformer-based 扩散/流匹配架构

### 硬件/数据相关
- [[NVIDIA A800]]: 训练硬件
- [[RoboFactory]]: 多机器人操作仿真环境

---

## 速查卡片

> [!summary] MultiWorld: Scalable Multi-Agent Multi-View Video World Models
> - **核心**: 统一框架实现多智能体动作可控 + 多视角几何一致的视频世界模型
> - **方法**: MACM（RoPE 智能体编码 + 自适应权重）+ GSE（冻结 VGGT 3D 特征 + Cross-Attention）
> - **结果**: 游戏场景 FVD 179（vs COMBO 207），RPE 0.67（vs 0.72）；机器人场景 FVD 96
> - **代码**: https://multi-world.github.io/

---

*笔记创建时间: 2026-04-24*
