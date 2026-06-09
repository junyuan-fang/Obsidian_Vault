---
title: "GN0: Toward a Unified Paradigm for Generation, Evaluation, and Policy Learning in Visual-Language Navigation"
method_name: "GN0"
authors: [Xinhai Li, Xiaotao Zhang, Yuehao Huang, Jiankun Dong, Tianhang Wang, Sunyao Zhou, Yunzi Wu, Chengnuo Sun, Yunfei Ge, Qizhen Weng, Chi Zhang, Chenjia Bai, Xuelong Li]
year: 2026
venue: arXiv
tags: [vision-language-navigation, embodied-ai, 3d-gaussian-splatting, navigation-foundation-model, dagger, grpo, bev-memory, dual-system]
zotero_collection: 5-具身智能与机器人/1-视觉语言导航
image_source: mixed
arxiv_html: https://arxiv.org/abs/2606.03682
created: 2026-06-04
---

# 论文笔记：GN0: Toward a Unified Paradigm for Generation, Evaluation, and Policy Learning in Visual-Language Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 中国电信人工智能研究院（TeleAI）+ 上海交大 / 浙大 / 同济 / 复旦 / 江苏大学 |
| 日期 | June 2026 |
| 项目主页 | https://telehuman-gn0.github.io |
| 代码 | https://github.com/TeleHuman/GN0 |
| 模型 | https://huggingface.co/TeleEmbodied/GN-BAE |
| 对比基线 | [[NaVid]], [[InternNav]], [[UniNaVid]], [[ETPNav]], [[StreamVLN]], [[DualVLN]], [[CorrectNav]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.03682) / [Code](https://github.com/TeleHuman/GN0) |

---

## 一句话总结

> GN0 用 3DGS 把 VLN 的数据生成、仿真评测、策略训练打通成一条流水线，并提出 SFT→DAgger→DAPO 三段式 RL 训练的导航基础模型 GN-BAE，在 GN-Bench 与 VLN-CE 上双双登顶。

---

## 核心贡献

1. **GN-Matrix 数据集**：基于 [[3D Gaussian Splatting]]（[[InteriorGS]] 1000 + 手工 150 + [[WorldGrow]] 扩展 2000 个场景）+ [[3DGS Dynamic Avatar|3DGS 动态化身]]，自动合成 47M 条导航轨迹，统一覆盖 goal-nav / instruction-following / human-following 三类任务。
2. **GN-Bench 仿真与基准**：首个引入 [[BEV Memory|BEV 度量评估]] 与动态 3DGS 化身的 [[Vision-Language Navigation|VLN]] 基准，把渲染、碰撞、A\* 规划、在线 relabel 整合进同一闭环交互器中，既能评测也能做 [[DAgger]] 数据采集。
3. **GN-BAE 导航基础模型**：以 [[Qwen3-VL]] 为骨干、采用 [[Trajectory Tokenization|轨迹分词]] 统一离散动作与连续 waypoint，提出 **SFT → DAgger → [[DAPO]] → [[NavDP|Action Expert]]** 的四阶段训练范式，并首次把 3DGS 渲染的 BEV 作为 [[BEV Memory|空间记忆]] 解锁 [[VLM]] 的隐式空间推理能力。
4. **Dual-System 部署**：高层 VLM 负责长程推理与轨迹规划，低层 [[NavDP]] [[Diffusion Policy]] 负责实时动作执行，Orin/H100 实测 5Hz 推理，无需任何真机数据即可 sim-to-real。

---

## 问题背景

### 要解决的问题

[[Vision-Language Navigation|VLN]] 系统长期受限于：(1) 高保真场景数据稀缺，mesh/全景图/视频序列难以兼顾真实感与可交互；(2) 评测停留在静态环境与离散全景图节点，无法刻画动态人机交互；(3) 主流方法是 [[SFT]] 端到端模仿，闭环执行时一旦偏离专家轨迹就崩，存在严重 [[Covariate Shift|协变量偏移]]。

### 现有方法的局限

- **离散 VLN**（R2R/R4R 全景图节点）剥离了几何与碰撞，无法部署。
- **连续 VLN**（VLN-CE / VLN-PE）依赖 Habitat + Matterport3D，视觉风格与真机相机差距大。
- **SFT-only 模型**（[[NaVid]]、[[UniNaVid]]、[[StreamVLN]]）只在专家轨迹的窄分布上学，遇到自身 rollout 误差就放大。
- **现有 3DGS 工作**（[[GaussNav]]、[[SAGE-Bench]]）做了感知或基准，但没把数据/仿真/训练/部署统一起来。

### 本文的动机

VLM 预训练于真实照片，3DGS 渲染最接近真实照片 → 用 3DGS 替代 mesh 仿真，可以最大化继承 VLM 的预训练知识；同时把仿真器本身做成"既能评测也能产数据"的闭环平台，配合 DAgger 把 SFT 后窄分布"打开"，再上 [[DAPO]] 做 RL 精炼，从根本上解决 covariate shift。

---

## 方法详解

### 总体架构

GN0 由三块组成：

- **GN-Matrix**：3DGS 场景资源 + 自动化轨迹合成 → 47M 训练数据
- **GN-Bench**：3DGS 闭环仿真器 + BEV 评测协议 + 动态人形化身
- **GN-BAE**：Dual-System 导航基础模型（VLM 规划器 + NavDP Action Expert 执行器）

### GN-Matrix 数据生成

**场景资源**：

| 来源 | 数量 | 平均面积 | 用途 |
|------|------|----------|------|
| [[InteriorGS]] | 1,000 | 150.1 m² | 高质量室内住宅 |
| 手工大场景 | 150 | 374.5 m² | 办公室 / 超市等大空间 |
| [[WorldGrow]] 6×6 | 1,760 | 91.7 m² | 文本扩展，提升多样性 |
| [[WorldGrow]] 8×8 | 240 | 158.8 m² | 大尺度长程导航 |

**动态化身**：用 SMPL-X 与 [[3DGS Dynamic Avatar|3DGS Avatar]] 绑定，从 Mixamo 拿 FBX 动作驱动，每帧用刚体变换更新 Gaussian 均值与协方差（见公式 5）。

**任务合成**：A\* 规划 + LLM 路径描述生成 goal-nav / instruction-following / human-following 三类 episode；每条轨迹同时输出 FPV、BEV、历史观测、6 步离散动作 + 12 点连续 waypoint。

### DAgger 数据采集（核心创新点之一）

为缓解 [[Covariate Shift|协变量偏移]]，在 SFT 之后让模型自己 rollout，再用 oracle 进行在线 relabel：

- **连续 waypoint 标签**：A\* 在膨胀占据图上求测地路径，投影回 ego 坐标
- **离散动作标签**：用 [[Model Predictive Control|MPC]] 跟踪 A\* 路径（公式 1）
- **停止逻辑**：进入成功圈后 10 步内未发 `<STOP>` 则强制覆写为 `<STOP>`（公式 2）

最终采集 **284K 闭环 DAgger 样本** + **22K DAPO 参考样本**。

### GN-Bench 仿真器

- **渲染**：基于 [[diff-gaussian-rasterization]]，并行渲染 FPV-RGB-D + 正交投影 BEV
- **状态**：$s_t = (p_t, \theta_t)$，混合动作空间 $\mathcal{A} = \mathcal{A}_{local} \cup \mathcal{A}_{global}$（公式 3）
- **碰撞**：占据图 $M$ 做圆形结构元 $B_r$ 形态膨胀得 $M_{dil} = M \oplus B_r$，路径用 Bresenham 算法离散，命中即截断（公式 4）
- **规划**：A\* 在膨胀图上跑，输出 oracle 路径与 SPL 真值
- **动态化身**：每个化身的 3DGS 用 $R_t, T_t$ 实时驱动（公式 5）

### GN-BAE 模型

**Backbone**：[[Qwen3-VL]]；输入 = 当前 FPV + 16 帧历史拼接图 + 可选 BEV；输出 = 离散动作序列 + 连续 BEV waypoint 序列。

**[[Trajectory Tokenization|轨迹分词]]**：
- 离散动作词表：`<action>`, `</action>`, `<FWD>`(0.25m), `<LEFT>`(15°), `<RIGHT>`(15°), `<STOP>`
- 连续 waypoint：BEV 坐标归一化后离散成 1000 桶，对应 `<0>`～`<999>`

**四阶段训练**：
1. **SFT**：vision tower lr=$2\times 10^{-6}$，LLM lr=$1\times 10^{-5}$，监督下 6 步离散动作 + 12 点 3m 轨迹（公式 6）
2. **DAgger**：在 284K 闭环样本上 5 epoch，lr=$1\times 10^{-6}$，目的不是降 loss 而是把策略"平铺"出来（公式 7）
3. **[[DAPO]]**：[[GRPO]] 风格相对优势估计（公式 8）+ clipped PG（公式 9-10）+ 结构化 reward（公式 11），group size $G=8$，**关闭 KL 正则**（避免破坏格式输出）
4. **[[NavDP|Action Expert]]**：VLM 出的 BEV 轨迹喂给 [[Diffusion Policy]] 做低层去噪，DDPM 训练（公式 12-13）+ critic + 跨模态对齐（公式 14）

---

## 关键公式

### 公式 1: [[Model Predictive Control|MPC 专家动作选择]]

$$
a_t^* = \arg\min_{a \in \mathcal{A}} \sum_{k=1}^{H} \gamma^k C(s_{t+k}, T_{bev,t}^*)
$$

**含义**：DAgger 数据采集阶段，oracle 用 MPC 在预测域 $H$ 内挑选离散动作以最小化对 A\* 真值轨迹的偏差。

**符号说明**：
- $H$: 预测域长度
- $\gamma$: 折扣因子
- $C(\cdot, \cdot)$: 当前状态相对最优 A\* 轨迹 $T^*_{bev,t}$ 的空间偏差代价
- $\mathcal{A}$: `{<FWD>, <LEFT>, <RIGHT>, <STOP>}`

### 公式 2: 强制停止启发式

$$
a_t^* = \texttt{<STOP>}, \quad \text{if } d(p_t, g) \le d_{success} \text{ and } c_{goal} \ge 10
$$

**含义**：进入成功圈 $d_{success}$ 后若 10 步内仍未自发停止，oracle 直接把终态标签覆写为 `<STOP>`，避免模型在目标附近 hover。

### 公式 3: 离散动作下的运动学更新

$$
s_{t+1} = \begin{cases}
\bigl(p_t + \Delta d \cdot [\cos\theta_t, \sin\theta_t]^\top,\, \theta_t\bigr), & a = \texttt{MoveForward} \\
\bigl(p_t,\, \theta_t \pm \Delta\theta\bigr), & a = \texttt{TurnLeft/Right}
\end{cases}
$$

**含义**：GN-Bench 的本地动作运动学。$\Delta d = 0.25\text{m}$，$\Delta\theta = 15°$，与 VLN-CE 兼容。

### 公式 4: 碰撞截断

$$
k = \max\{i \in [0, N] \mid M_{dil}(p_j) = 0,\ \forall j \le i\}
$$

**含义**：从 $p_{src}$ 到 $p_{dst}$ 用 Bresenham 离散成像素链，遇到 $M_{dil}$ 中障碍则截断，最终位置为 $p_k$。

### 公式 5: [[3DGS Dynamic Avatar|3DGS 化身的刚体驱动]]

$$
\mu_i^{(t)} = R_t \mu_i + T_t, \qquad \Sigma_i^{(t)} = R_t \Sigma_i R_t^\top
$$

**含义**：把人形化身上每个 Gaussian 的均值/协方差按时间相关旋转 $R_t$ 与平移 $T_t$ 变换，再与静态场景 $G_{static}$ 拼接渲染。

**符号说明**：
- $\mu_i, \Sigma_i$: 第 $i$ 个高斯点的均值与协方差
- $R_t \in SO(3), T_t \in \mathbb{R}^3$: 第 $t$ 帧的 6-DoF 姿态

### 公式 6: [[SFT|监督微调损失]]

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y^*)\sim \mathcal{D}} \left[ \frac{1}{L}\sum_{t=1}^{L} \log \pi_\theta(y_t^* \mid y_{<t}^*, x) \right]
$$

**含义**：标准自回归 cross-entropy，对 trajectory token 序列做 teacher forcing。

### 公式 7: [[DAgger]] 自适应损失

$$
\mathcal{L}_{DAgger}(\theta) = -\mathbb{E}_{(x,y^*)\sim \mathcal{D}_{DAgger}} \left[ \frac{1}{L}\sum_{t=1}^{L} \log \pi_\theta(y_t^* \mid y_{<t}^*, x) \right]
$$

**含义**：形式同 SFT 损失，但训练数据 $\mathcal{D}_{DAgger}$ 来自策略自己的闭环 rollout + oracle relabel，把策略从专家窄分布"打开"。

### 公式 8: [[GRPO]] 组内相对优势

$$
A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

**含义**：每条 prompt 采样 $G=8$ 条候选轨迹，组内 z-score 归一化得到相对优势，无需独立 value head。

### 公式 9-10: [[DAPO]] clipped 策略梯度

$$
\mathcal{J}_{DAPO}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\bigl(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon_{low}, 1+\epsilon_{high}) A_i\bigr)
$$

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} \mid x, y_{i,<t})}
$$

**含义**：PPO 风格的 ratio clipping，但用 GRPO 的组归一化优势 + 非对称 clipping 边界。实现中 **关闭 KL** 因为格式约束已由 reward 显式惩罚保证。

### 公式 11: [[DAPO]] 执行级 reward

$$
R(\hat{a}, a^*) = R_{traj} + \lambda_{first} R_{first} - \lambda_{col} R_{col} - \lambda_{len} R_{len} - \lambda_{fmt} R_{fmt}
$$

**含义**：综合 rollout 轨迹一致性 ($R_{traj}$，位置+朝向)、首动作正确性 ($R_{first}$)、碰撞惩罚 ($R_{col}$)、长度失配惩罚 ($R_{len}$)、格式惩罚 ($R_{fmt}$)。最后两项确保模型不破坏结构化输出。

### 公式 12-13: [[Diffusion Policy|Action Expert]] 扩散损失

$$
a_\tau = \sqrt{\bar{\alpha}_\tau}\, a + \sqrt{1-\bar{\alpha}_\tau}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

$$
\mathcal{L}_{act} = \mathbb{E}\bigl[\|\hat{\epsilon}_\theta(a_\tau, \tau, \cdot) - \epsilon\|_2^2\bigr]
$$

**含义**：[[NavDP]] Action Expert 采用 [[DDPM]] 噪声预测目标，输入仅 RGB + 目标条件（去掉了原版的 depth 输入）。

### 公式 14: Action Expert 总损失

$$
\mathcal{L} = 0.8\, \mathcal{L}_{act} + 0.2\, \mathcal{L}_{critic} + 0.5\, \mathcal{L}_{aux}
$$

**含义**：扩散动作损失 + 障碍感知的 critic 回归 + 跨模态目标对齐辅助损失，权重经验设置。

---

## 关键图表

### Figure 1: GN0 总览

![[GN0_fig1_overview.png]]

**说明**：GN0 三大组件 GN-Matrix（数据）/ GN-Bench（仿真）/ GN-BAE（模型）的总览图，体现"数据 → 仿真 → 训练 → 部署"的统一闭环。

### Figure 2: GN-Matrix 数据生成 Pipeline

![[GN0_fig2_matrix_pipeline.png]]

**说明**：多源 3DGS 场景（[[InteriorGS]] + 手工 + [[WorldGrow]]）+ 动态化身 → 任务合成（goal/instruction/human-following）→ 多模态标注（FPV + BEV + 历史 + 动作 + waypoint）。

### Figure 5: GN-Bench 仿真器总览

![[GN0_fig5_bench_overview.png]]

**说明**：GN-Bench 高保真闭环交互能力示意。同时展示了 FPV RGB-D 流、BEV 正交投影、动态化身、A\* 规划与碰撞处理一体化的设计。

### Figure 6: BAE 架构与四阶段训练

![[GN0_fig6_bae_architecture.png]]

**说明**：BAE 模型架构与 SFT → DAgger → DAPO → Action Expert 四阶段训练 pipeline。VLM 上层做轨迹规划，下层 [[NavDP]] [[Diffusion Policy]] 做执行。

### Figure 7: 真机部署

![[GN0_fig7_realworld.png]]

**说明**：在自研轮臂机器人 TeleBotW、自研小型人形 TeleBotM、宇树 G1 上做的三段任务（暗室穿越、走廊跟随幕布、长走廊多门拐弯），全程零真机数据训练。

### Table 1: 基准兼容性与数据集输出

| Setting | Instruction Format | Action Space | Notes |
|---------|--------------------|--------------|-------|
| Native GN-Matrix | Goal / instruction / human-follow | Path / action supervision | 3DGS 原生表征 |
| VLN-CE 兼容 | R2R 风格自然语言 | 离散动作 `<FWD/LEFT/RIGHT/STOP>` | 适配 Habitat |
| VLN-PE 兼容 | 同上 | 含物理动力学 | Unitree H1 |

**说明**：GN-Matrix 既可作为原生 3DGS 数据集，也可适配现有 [[VLN-CE]] / [[VLN-PE]] 控制约定。

### Table 2: GN-Bench 多模态观测空间

| Modality | Camera | Tensor Shape | FoV / Scale | 主要用途 |
|----------|--------|--------------|-------------|----------|
| Egocentric RGB | Perspective | $H \times W \times 3$ | 可配置 | VLM 视觉语义推理 |
| Egocentric Depth | Perspective | $H \times W \times 1$ | 可配置 | 局部几何与避障 |
| Top-down BEV | Orthographic | $H_{bev} \times W_{bev} \times C$ | Metric-scale | 全局空间先验 / 拓扑规划 |

### Table 3: GN-Bench 主结果（Unseen split, SR/SPL）

| Method | Depth | BEV | FPV | NE↓ | OS↑ | SR↑ | SPL↑ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CMA | ✓ | | ✓ | 8.1 | 19.6 | 15.5 | 14.9 |
| NaVid | | | ✓ | 7.7 | 20.3 | 14.5 | 12.8 |
| UniNaVid | | | ✓ | 7.8 | 20.7 | 12.8 | 10.3 |
| InternNav(S2) | | | ✓ | 7.2 | 26.7 | 22.1 | 20.3 |
| NaVid† (SFT on GN-Matrix) | | | ✓ | 7.1 | 23.8 | 23.1 | 23.0 |
| UniNaVid† | | | ✓ | 7.5 | 23.1 | 20.8 | 20.2 |
| InternNav(S2)† | | | ✓ | 6.9 | 24.9 | 24.0 | 23.7 |
| **GN-BAE (FPV)** | | | ✓ | **5.6** | **43.6** | **38.9** | **37.3** |
| **GN-BAE (BEV+FPV)** | | ✓ | ✓ | 5.8 | 40.2 | 38.5 | **38.2** |

**说明**：仅用单目 RGB，GN-BAE 在 Unseen 上 SR=38.9，比最强微调 baseline InternNav-S2† 高 14.9 点。BEV 在 Seen 上更强（SR=58.6），在 Unseen 反而略低，说明陌生场景下 BEV 重建噪声会拖累长程决策。

### Table 4: VLN-CE R2R Val-Unseen 对比

| Method | 输入 | NE↓ | OS↑ | SR↑ | SPL↑ |
|--------|------|:---:|:---:|:---:|:---:|
| ETPNav* | Pano+Odo+Depth | 4.71 | 65.0 | 57.0 | 49.0 |
| NaVILA | Single RGB | 5.22 | 62.5 | 54.0 | 49.0 |
| StreamVLN | Single RGB | 4.98 | 64.2 | 56.9 | 51.9 |
| DualVLN | Single RGB | 4.05 | 70.7 | 64.3 | 58.5 |
| CorrectNav | Single RGB | 4.24 | 67.5 | 65.1 | 62.3 |
| **GN-BAE** | Single RGB | **3.50** | **75.6** | **67.7** | **63.4** |

**说明**：纯单目 RGB 设置下 NE 比 CorrectNav 降 17.5%，SR/SPL 同时领先，验证 sim-to-VLN-CE 的迁移能力。

### Table 5: 消融实验（Full Pipeline vs w/o stages）

| Method | Seen SR↑ | Seen SPL↑ | Unseen SR↑ | Unseen SPL↑ |
|--------|:---:|:---:|:---:|:---:|
| **Full (FPV)** | **46.9** | **45.1** | **38.9** | **37.3** |
| w/o DAgger | 26.5 | 25.8 | 37.8 | 36.0 |
| w/o DAPO | 41.9 | 40.2 | 28.1 | 27.3 |
| w/o DAgger & DAPO | 26.1 | 24.8 | 29.4 | 28.1 |
| **Full (BEV+FPV)** | **58.7** | **58.6** | **38.5** | **38.2** |
| w/o DAgger | 25.6 | 25.0 | 32.4 | 32.3 |
| w/o DAPO | 47.6 | 47.4 | 27.8 | 26.9 |

**关键发现**：
- **DAgger 是核心**：去掉后 Seen SR 直接腰斩（46.9→26.5），证明窄分布问题致命
- **DAPO 拉效率**：去掉后 SPL 与 Unseen SR 显著下降
- **直接跳过 DAgger 上 RL 几乎不涨**：因为 SFT 后策略熵太低，DAPO 无可探索空间

### Table 6: GN-Matrix 场景源细分

| Source | # Scenes | 平均面积 (m²) | 平均物体数 |
|--------|---------:|---------:|---------:|
| InteriorGS | 1,000 | 150.1 | 502.9 |
| 手工大场景 | 150 | 374.5 | 2321.4 |
| WorldGrow 6×6 | 1,760 | 91.7 | N/A |
| WorldGrow 8×8 | 240 | 158.8 | N/A |

---

## 实验

### 数据集与基准

| 平台 | 用途 |
|------|------|
| **GN-Bench** | 主要 3DGS 高保真连续导航环境 |
| **VLN-CE** (R2R Val-Unseen) | 跨平台泛化测试，Habitat + Matterport3D |
| **VLN-PE** | 含真机动力学，Unitree H1 仿真 |
| 真机：TeleBotW / TeleBotM / Unitree G1 | NVIDIA Orin 端侧 + H100 推理，5Hz |

### 实现细节

- **Backbone**: [[Qwen3-VL]]
- **SFT**: vision tower lr $2\times 10^{-6}$，LLM/projector lr $1\times 10^{-5}$
- **DAgger**: 284K 样本，5 epoch，lr $1\times 10^{-6}$
- **DAPO**: lr $1\times 10^{-6}$，group $G=8$，batch=64，2 epoch，**关闭 KL**
- **Action Expert**: [[NavDP]]，移除 depth 输入，DDPM 训练，分 follow / planning 两个专家
- **数据规模**: 47M 总轨迹 + 22K DAPO 参考样本

### 可视化结果

真机段落展示 GN-BAE 仅用 GS 仿真数据训练即可：暗室直行 + 右转出门 → 走廊跟左侧幕布 + 拐角右转 → 长走廊穿水机 + 右转拐弯，三段任务均稳定完成。

---

## 批判性思考

### 优点

1. **完整闭环**：数据 → 仿真 → 训练 → 部署 一条龙打通，是目前最系统的 3DGS-VLN 方案。
2. **DAgger 设计有理**：消融数据非常硬，"先 DAgger 打开窄分布再 RL"是 SFT→RL 范式上常被忽视的关键中间步。
3. **轨迹分词巧妙**：把 BEV waypoint 离散成 1000 token 让自回归 VLM 输出连续轨迹，对训练效率与多任务统一非常友好。
4. **真机零迁移**：纯仿真训出来的策略能直接跑 Unitree G1，说明 3DGS 渲染确实弥合了 sim-to-real 视觉 gap。

### 局限性

1. **BEV 在 Unseen 反而拖后腿**：说明 3DGS BEV 在分布外重建质量不稳，作为"地图"用还有距离。
2. **依赖 oracle A\*+MPC**：DAgger 数据全靠规划器，复杂语义指令（"找一个有沙发的房间"）oracle 难以提供精准标签。
3. **47M 数据 + 大量 H100 训练**：复现门槛极高，小团队几乎不可能跟。
4. **动态化身只走脚本动画**：所谓 HRI 评估实际是"被动避人"，没有真正的交互反馈循环。
5. **关闭 KL 是 trick**：用 reward 显式惩罚格式来替代 KL 约束，对其他下游任务的 RL 设计是否普适存疑。

### 潜在改进方向

1. 用 [[World Model|世界模型]] 替代 oracle 做 DAgger relabel，摆脱对 A\* 的依赖。
2. 把 BEV memory 升级为可学习的稀疏 3D representation（如 NeRF/3DGS tokens）。
3. 引入真正可交互的动态化身（行为模型 + 反应式动画），让 HRI 评估更靠谱。
4. 探索小模型 + 知识蒸馏路径，降低 5Hz H100 推理门槛。

### 可复现性评估

- [x] 代码开源（GitHub）
- [x] 预训练模型（HuggingFace）
- [x] 训练细节相对完整（学习率、batch、epoch 全列）
- [ ] 47M 数据完全开源情况未知
- [ ] 自研机器人不可复现

---

## 关联笔记

### 基于
- [[3D Gaussian Splatting]]：渲染基础
- [[InteriorGS]]：基底场景资产
- [[WorldGrow]]：场景扩展
- [[Qwen3-VL]]：VLM backbone
- [[NavDP]]：Action Expert 底层

### 对比
- [[NaVid]] / [[UniNaVid]]：纯 SFT VLM 导航模型
- [[InternNav]]：另一条 dual-system 路线
- [[StreamVLN]] / [[DualVLN]] / [[CorrectNav]]：VLN-CE 上的强 RGB-only baseline
- [[GaussNav]] / [[SAGE-Bench]]：同样基于 3DGS 但只解决感知或基准

### 方法相关
- [[DAgger]]：核心闭环适配阶段
- [[DAPO]] / [[GRPO]]：RL 优化目标
- [[Trajectory Tokenization]]：连续动作离散化的关键设计
- [[BEV Memory]]：本文首次形式化的空间记忆机制
- [[Diffusion Policy]]：Action Expert 范式

### 硬件/数据相关
- [[VLN-CE]] / [[VLN-PE]]：评测协议
- [[Vision-Language Navigation]]：任务范式

---

## 速查卡片

> [!summary] GN0: A Unified Paradigm for VLN
> - **核心**：用 3DGS 把 VLN 数据/仿真/训练/部署打通成一条流水线
> - **方法**：GN-Matrix (47M 数据) + GN-Bench (3DGS 闭环仿真+BEV评测) + GN-BAE (SFT→DAgger→DAPO→NavDP 四段式)
> - **结果**：GN-Bench Unseen SR 38.9（+15 vs SOTA），VLN-CE R2R Val-Unseen SR 67.7 / SPL 63.4
> - **代码**：https://github.com/TeleHuman/GN0

---

*笔记创建时间: 2026-06-04*
