---
title: "STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images"
method_name: "STRIPS-WM"
authors: [Abhiroop Ajith, Constantinos Chamzas]
year: 2026
venue: arXiv
tags: [world-model, neuro-symbolic-planning, strips, visual-planning, propositional-planning, fsq, task-graph]
zotero_collection: 1-世界模型与视频生成
image_source: online
arxiv_html: https://arxiv.org/html/2606.06832v1
created: 2026-06-09
---

# 论文笔记：STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Worcester Polytechnic Institute (WPI) |
| 日期 | June 2026 |
| 项目主页 | 无 |
| 对比基线 | [[LatPlan]] (AMA3) / [[Latent Space Roadmap]] (LSR) |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06832) / [HTML](https://arxiv.org/html/2606.06832v1) / Code 未开源 |

---

## 一句话总结

> 从纯图像转移数据中端到端学出**经典 STRIPS 符号规划模型**（谓词、前置条件、增删效果），实现可被传统规划器（BFS/A\*）直接调用的视觉任务规划。

---

## 核心贡献

1. **三阶段神经-符号管线**: Image → [[Image-Grounded Task Graph|抽象转移图]] → [[STRIPS]] 算子 → 视觉 [[Propositional Planning|命题]] 分类器，把"从像素学规划"问题拆成可独立优化的三段
2. **CP-SAT 化的算子合成**: 把 [[STRIPS]] 算子学习写成布尔约束满足问题，用 Google **CP-SAT** 在 300 s 预算内同时求解二值谓词向量、precondition mask、add/del mask，并通过松弛变量 $\xi, \eta$ 容忍噪声转移和不完整动作覆盖
3. **可证明的正确性**: 当 $\xi=\eta=0$ 时学出的 STRIPS 模型对学习图是 **sound & complete**（Theorem 10），并保证 add/del/precondition mask 互斥（Lemma 1）
4. **优于现有视觉规划基线**: 在 [[Visual Rearrangement Task|视觉重排]] 三个域（BlocksWorld / DinnerTable sim / DinnerTable Real）上，长 horizon 时显著超越 [[LatPlan]]-AMA3、[[Latent Space Roadmap]] 与 latent rollout/BFS 基线

---

## 问题背景

### 要解决的问题

机器人接收一张当前 RGB 图和一张目标 RGB 图，需要输出一串高层动作 $a_1, ..., a_T$ 让真实世界从当前态走到目标态——即 **image-to-plan**。问题难点是：**规划需要离散符号结构（谓词、前置/效果）**，但输入是高维连续像素。

### 现有方法的局限

论文把已有视觉规划分三类，每一类都有结构性缺陷：

1. **Reconstruction-based [[Video World Model|视频世界模型]]**（如 Dreamer 系）: 在像素/特征空间 rollout，长 horizon 误差爆炸，不能调用经典规划器
2. **Latent-space search**（[[Latent Space Roadmap]] LSR）: 把图像 embed 后构图搜索，**没有显式算子结构**，无法泛化到训练中未观察过的状态对组合
3. **Neuro-symbolic 抽象**（[[LatPlan]]-AMA3）: 已经尝试学 STRIPS 谓词，但端到端 VAE-style 学到的 propositional code 与可被规划器使用的 precondition/effect 结构不一致，长 horizon 失败率高

### 本文的动机

经典 [[STRIPS]] 规划器（A\*, BFS over symbolic state）在低维离散状态上极其高效，**只要状态、算子是正确的**。所以核心瓶颈不是规划算法，而是**怎么从图像里把谓词和算子"挖"出来**。本文把这件事拆成两步：先无监督学一个状态聚类（抽象转移图），再把算子学习写成可证明的 CP-SAT。

---

## 方法详解

### 总体架构

STRIPS-WM 采用 **三阶段流水线**，每段独立训练，前段输出被后段当作标签：

1. **Stage 1 — [[Image-Grounded Task Graph]]**: 用 [[Joint Embedding Predictive Architecture|JEPA]] 风格自监督把每张图编码成离散 [[FSQ]] code $z_t \in \mathbb{Z}^d$，得到一张抽象图 $\hat{G} = (\hat{S}, \hat{E})$，节点是 code，边是观察到的 $(z_t, a_t, z_{t+1})$
2. **Stage 2 — [[STRIPS]] 算子学习**: 在 $\hat{G}$ 上跑一次 **[[CP-SAT|布尔约束求解]]**，同时输出：
   - 每个抽象状态 $\hat{s}$ 的 $m$ 维谓词向量 $x_{\hat{s}} \in \{0,1\}^m$
   - 每个动作 $a$ 的 (precondition$^+$, precondition$^-$, add, delete) mask
3. **Stage 3 — 视觉 [[Propositional Planning|命题]] 分类器**: 用 [[ConvNeXt]]-Tiny 学一个 image → $m$ 维 0/1 向量的映射，测试时在线把任意 RGB 图二值化成谓词，喂给经典 [[A* Search|A\*]]/BFS 规划器

**关键设计**: 这三段共享同一个谓词向量空间 $\{0,1\}^m$，所以 Stage 1 学到的状态等价类可以直接被 Stage 2 的算子操作，Stage 3 又把这套表示泛化到任意新图。

### 模块 1: Image-Grounded Task Graph (Stage 1)

**设计动机**: 不依赖任何符号监督，只用 $(o_t, a_t, o_{t+1})$ 三元组学一个**离散、动作可分**的状态嵌入。

**具体实现**:

- **Student 编码器** $f_\theta$: ResNet → projection → [[FSQ|Finite Scalar Quantization]] 量化到 $[5]^d$ 网格（$d \in \{5,6,7\}$）。BlocksWorld 用 $[5]^5$，DinnerTable 系列用 $[5]^7$
- **EMA Teacher** $f_{\bar\theta}$: 动量 $\mu = 0.996$ 更新，提供 successor 目标 code $z^{\text{tar}}_{t+1}$
- **Latent 动力学** $g_\phi$: 接 $(z_t, a_t)$ 预测 $\hat z_{t+1}$，与 teacher target 比对（[[JEPA]] 风格）
- **Inverse Model**: 接 $(z_t, z_{t+1})$ 预测 $\hat a_t$，强迫 code 保留区分动作的信息（防止表征塌缩到 trivial 常量）
- **Predictive Separation 损失**: 对 latent nondeterminism 做正则——当两条样本 $(z_t, a_t)$ 相同但 successor 不同时，强迫它们的 pre-quantization latent 拉远，减少 [[Visual Aliasing|视觉混叠]]

**输出**: 离散状态分配函数 $\alpha: \mathcal{O} \to \hat{\mathcal{S}}$ 和观察到的边集 $\hat{E}$。

### 模块 2: STRIPS 算子学习 (Stage 2)

**设计动机**: 把"找一组谓词 + 算子，使所有观察到的 transition 都可解释"这件事，写成可以被现成求解器精确解决的优化问题。

**变量** (全部 0/1):

- $x_{\hat{s},p}$: 抽象状态 $\hat{s}$ 的第 $p$ 个 [[Propositional Predicate|谓词]] 值，$p \in \{1,...,m\}$
- $\text{pre}^+_{a,p}, \text{pre}^-_{a,p}$: 动作 $a$ 对谓词 $p$ 的正/负前置条件
- $\text{add}_{a,p}, \text{del}_{a,p}$: 动作 $a$ 对谓词 $p$ 的增/删效果
- $\xi_{\hat{e},p}$: 观察边 $\hat{e}$ 在谓词 $p$ 上的**转移松弛变量**
- $\eta_{\hat{s},a}$: trusted missing pair $(\hat{s}, a)$ 的**可适用性松弛变量**

**核心约束**:

- **(C1) 前置满足**: $\text{pre}^+_a \le x_{\hat{s}}, \quad \text{pre}^-_a \le 1 - x_{\hat{s}}$
- **(C2) 效果一致**: $\tau_{\hat\omega_a}(x_{\hat{s}_0}) = x_{\hat{s}_1}$，其中 [[STRIPS|STRIPS 转移函数]] 为 $x_{\sigma_1} = \text{add}_a \vee (x_{\sigma_0} \wedge \neg \text{del}_a)$
- **(C3) Negative Evidence**: 对 [[Trusted Missing Pair|trusted missing pair]] $(\hat{s}, a) \in \hat{N}$，强制 $\hat\omega_a$ 在 $x_{\hat{s}}$ 上**不可适用**

**Trusted missing pair**: 关键概念。如果某个状态 $\hat{s}$ 出度已经接近完整（观察过大部分可能动作），那么**未观察到**的 $(\hat{s}, a)$ 就被视为"动作 $a$ 在此处不可行"的强证据（局部 [[Closed-World Assumption|闭世界假设]]），为前置条件提供负样本。

### 模块 3: 视觉 Predicate Classifier (Stage 3)

**设计动机**: Stage 2 输出的谓词只定义在训练观察到的有限离散状态上。要把这套符号表示扩展到任意新图，需要一个**像素 → 谓词**的视觉模型。

**具体实现**:

- **Backbone**: [[ConvNeXt]]-Tiny，输入 $228 \times 228$ RGB
- **输出头**: $m$ 维 sigmoid，预测每个谓词为 1 的概率
- **训练标签**: 来自 Stage 2 的 $x_{\hat{s}} \in \{0,1\}^m$
- **损失**: [[Binary Cross-Entropy]] $\mathcal{L}_{\text{pred}} = \sum_p \text{BCE}(\hat p_p, x_{\hat{s},p})$
- **测试时**: 0.5 阈值二值化 → 喂给经典 [[A* Search|A\*]]/BFS 规划器 → 输出动作序列

---

## 关键公式

### 公式 1: [[FSQ|Finite Scalar Quantization]] 状态编码

$$
z_t = q(f_\theta(o_t)), \quad z^{\text{tar}}_{t+1} = q(f_{\bar\theta}(o_{t+1}))
$$

**含义**: Student 和 teacher 编码器把图像投到欧式空间后，用 [[FSQ]] 把每一维 round 到 5 个固定 level 之一，得到离散 code。

**符号说明**:
- $f_\theta, f_{\bar\theta}$: student / EMA teacher 编码器
- $q(\cdot)$: [[FSQ]] round-to-nearest 量化
- $z_t \in [5]^d$: 离散状态码，$d \in \{5, 6, 7\}$

### 公式 2: [[JEPA|JEPA-style]] 动力学损失

$$
\mathcal{L}_{\text{dyn}} = \| \hat{z}_{t+1} - z^{\text{tar}}_{t+1} \|_2^2
$$

**含义**: 在 code 空间预测下一步，避免像素级重构。

**符号说明**:
- $\hat{z}_{t+1} = g_\phi(z_t, a_t)$: 学习的动力学模型预测
- $z^{\text{tar}}_{t+1}$: teacher 给出的真实 successor code（stop-gradient）

### 公式 3: [[Inverse Dynamics]] 损失

$$
\mathcal{L}_{\text{inv}} = \text{CE}(\hat{a}_t, a_t)
$$

**含义**: 用 $(z_t, z_{t+1})$ 反推动作，强迫 code 保留动作区分性。

**符号说明**:
- $\hat{a}_t$: 反向模型预测的动作分布
- $\text{CE}$: 交叉熵

### 公式 4: [[EMA|EMA Teacher 更新]]

$$
\bar\theta \leftarrow \mu \bar\theta + (1 - \mu) \theta, \quad \mu = 0.996
$$

**含义**: Teacher 参数是 student 的滑动平均，保证目标稳定。

### 公式 5: [[STRIPS|STRIPS 转移函数]]

$$
x_{\sigma_1} = \tau_{\omega_a}(x_{\sigma_0}) = \text{add}_a \vee (x_{\sigma_0} \wedge \neg \text{del}_a)
$$

**含义**: 经典命题 STRIPS 语义——后继状态 = (前态 ∧ 没被删) ∨ 新加。

**符号说明**:
- $x_{\sigma} \in \{0,1\}^m$: 状态的谓词向量
- $\text{add}_a, \text{del}_a \in \{0,1\}^m$: 动作 $a$ 的增/删 mask
- $\vee, \wedge, \neg$: 按位逻辑运算

### 公式 6: 观察边约束（Eq. 4–7）

$$
\begin{aligned}
\hat{x}'_p + \xi_{\hat{e},p} &\ge \text{add}_{a,p} \\
\hat{x}'_p + \text{del}_{a,p} &\le 1 + \xi_{\hat{e},p} \\
\hat{x}'_p - \hat{x}_p &\le \text{add}_{a,p} + \text{del}_{a,p} + \xi_{\hat{e},p} \\
\hat{x}_p - \hat{x}'_p &\le \text{add}_{a,p} + \text{del}_{a,p} + \xi_{\hat{e},p}
\end{aligned}
$$

**含义**: 用 4 条线性不等式精确编码 (C2) "效果可解释这条观察边"，并通过 $\xi$ 允许少量违反。

**符号说明**:
- $\hat{x}, \hat{x}'$: 边的前后状态谓词
- $\xi_{\hat{e},p} \in \{0,1\}$: 该边、该谓词上的松弛量；目标里第一优先级最小化

### 公式 7: [[Mask Disjointness]] 约束（Eq. 8）

$$
\text{add}_{a,p} + \text{del}_{a,p} \le 1, \quad \text{pre}^+_{a,p} + \text{pre}^-_{a,p} \le 1
$$

**含义**: 同一谓词不能同时被一个动作"加 + 删"，也不能同时要求"为真 + 为假"。Lemma 1 由此约束保证 well-formed STRIPS 算子。

### 公式 8: Negative Evidence 约束（Eq. 11）

$$
\sum_{p=1}^m \left( v^+_{\hat{s},a,p} + v^-_{\hat{s},a,p} \right) + \eta_{\hat{s},a} \ge 1, \quad \forall (\hat{s},a) \in \hat{N}
$$

**含义**: 对每个 trusted missing pair，要求至少存在一个**违反前置**的谓词（即 $a$ 在 $\hat{s}$ 真的不可适用）；否则 $\eta$ 付出代价。

**符号说明**:
- $v^+, v^-$ 表示动作 $a$ 在状态 $\hat{s}$ 上某个正/负前置不被满足
- $\hat{N}$: trusted missing pair 集合

### 公式 9: 字典序目标（Eq. 1，主目标）

$$
\min\nolimits_{\text{lex}} \left( \sum_{\hat{e} \in \hat{E}} \sum_{p=1}^m \xi_{\hat{e},p}, \quad \sum_{(\hat{s},a) \in \hat{N}} \eta_{\hat{s},a}, \quad \sum_{a \in A} \sum_{p=1}^m (\text{add}_{a,p} + \text{del}_{a,p}), \quad \sum_{a \in A} \sum_{p=1}^m (\text{pre}^+_{a,p} + \text{pre}^-_{a,p}) \right)
$$

**含义**: **字典序**四层优先级——先优先把转移松弛 $\sum\xi$ 压到 0（保真），再最小化负证据违反 $\sum\eta$，再压效果稀疏度，最后压前置稀疏度。先抓正确性、再抓 [[Occam's Razor|奥卡姆剃刀]] 式的简洁解。

### 公式 10: [[Binary Cross-Entropy|视觉谓词分类]] 损失

$$
\mathcal{L}_{\text{pred}} = \sum_{p=1}^m \text{BCE}\!\left( \hat{p}_p(o), x_{\hat\alpha(o), p} \right)
$$

**含义**: Stage 3 训练 [[ConvNeXt]] 用 Stage 2 学到的谓词向量作为监督。

---

## 关键图表

### Figure 1: Three Planning Spaces / 三种规划空间对比

![Figure 1](https://arxiv.org/html/2606.06832v1/x1.png)

**说明**: 同一视觉重排任务在三种空间里规划的对比。

- **顶部 — 图像空间**: 机器人观察 RGB 图，边是高层动作 ID。直接搜索像素状态不可行（维度爆炸）
- **中部 — 抽象任务图** $\hat{G}$: Stage 1 学到的离散 [[FSQ]] code 节点 + 观察到的转移边。可以做图搜索但**没有算子结构**，看不到新状态对的可达性
- **底部 — STRIPS 空间**: Stage 2 学到的 $m$ 维 0/1 谓词空间 + 显式算子 $(\text{pre}, \text{add}, \text{del})$。可被经典规划器使用，组合新目标也能搜

### Figure 2: Image-Grounded Task Graph 架构

![Figure 2](https://arxiv.org/html/2606.06832v1/x2.png)

**说明**: Stage 1 的完整训练图。

- **左路 — Student**: $o_t \to f_\theta \to z_t$（[[FSQ]] 量化）
- **右路 — Teacher**: $o_{t+1} \to f_{\bar\theta} \to z^{\text{tar}}_{t+1}$（EMA 复制 student，stop-gradient）
- **中间 — Action-Conditioned Predictor** $g_\phi$: 接 $(z_t, a_t)$ 预测 $\hat z_{t+1}$，与 teacher target 算 $\mathcal{L}_{\text{dyn}}$
- **下路 — [[Inverse Dynamics|Inverse Model]]**: 接 $(z_t, z_{t+1})$ 预测 $\hat a_t$，算 $\mathcal{L}_{\text{inv}}$
- **可选 — Predictive Separation Head**: 检测到 latent nondeterminism 时增加分离损失

整体训练目标 $\mathcal{L} = \mathcal{L}_{\text{dyn}} + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}} + \lambda_{\text{sep}} \mathcal{L}_{\text{sep}}$。

### Figure 3: 规划成功率 vs Horizon

![Figure 3](https://arxiv.org/html/2606.06832v1/x3.png)

**说明**: 三个域上规划成功率随**最短路 horizon** 的变化曲线。横轴是 ground-truth 最短路径步数（1 到 8+），纵轴是成功率。

- **STRIPS-WM**: 在所有 horizon 上几乎都 ≈ 100%
- **WM-Rollout / WM-BFS**: 在 horizon ≥ 4 后急剧下跌，因为 latent 动力学误差累积
- **[[LatPlan]]-AMA3**: 短 horizon 还可以，长 horizon 因前置/效果不一致而失败
- **[[Latent Space Roadmap|LSR]]**: 对新目标状态泛化差

### Figure 4: DinnerTable Real 域上 STRIPS 模型 vs 直接图搜索

![Figure 4](https://arxiv.org/html/2606.06832v1/x4.png)

**说明**: 同样用 Stage 1 学到的 $\hat{G}$ 做规划，对比"直接在 $\hat{G}$ 上 BFS 图搜索" vs "用 STRIPS-WM 跑完 Stage 2/3 后用经典规划器"。

**关键发现**: 即使两者用同一份观察数据，**学出 STRIPS 算子后能泛化到训练中没见过的状态对组合**——直接图搜索碰到新的 $(start, goal)$ 对时无路径，但 STRIPS 通过谓词组合可以搜出新路径。

### Table 1: 域统计与抽象指标

| Domain | $|A|$ | $|\mathcal{D}_O|$ | $|S^\star|$ | $|\hat{S}|$ | $|\hat{X}|$ | $m$ | $\sum\xi$ | $\sum\eta$ |
|---|---|---|---|---|---|---|---|---|
| BlocksWorld | 18 | 5,000 | 16 | 16 | 16 | 9 | **0** | **0** |
| DinnerTable | 70 | 12,000 | 101 | 101 | 101 | 35 | 0 | 9 |
| DinnerTable Real | 64 | 3,000 | 71 | 111 | 71 | 35 | 0 | 13 |

**说明**:
- $|A|$ 动作数, $|\mathcal{D}_O|$ 观察转移数, $|S^\star|$ 真实任务状态数, $|\hat{S}|$ 抽象 code 数, $|\hat{X}|$ 谓词去重后状态数, $m$ 谓词维度
- **BlocksWorld**: 恰好恢复 9 谓词的精确 STRIPS 模型（与人类工程化的版本一致），松弛 = 0
- **DinnerTable Sim**: 视觉抽象一一对应真实状态，但有 9 单位 negative evidence 松弛（部分动作覆盖不全）
- **DinnerTable Real**: Stage 1 的视觉 code 过度切分（111 vs 71），Stage 2 的谓词学习把它**合并回 71 个语义状态**——证明 STRIPS 抽象具备纠正过分割的能力

### Table 2: Task Graph 超参（节选）

| 项目 | BlocksWorld | DinnerTable | DinnerTable Real |
|---|---|---|---|
| FSQ levels | $[5]^5$ | $[5]^7$ | $[5]^7$ |
| EMA momentum $\mu$ | 0.996 | 0.996 | 0.996 |
| Backbone | ResNet | ResNet | ResNet |
| Image res | 228 × 228 | 228 × 228 | 228 × 228 |

### Table 3: CP-SAT 求解器设置

| 项目 | 取值 |
|---|---|
| Solver | Google OR-Tools CP-SAT |
| Wall-clock budget | 300 s per fixed-$m$ run |
| $m$ sweep | 二分搜索找最小可行 $m$ |
| 目标函数 | 字典序 4 层（Eq. 1） |

### Table 4: 视觉 Predicate Classifier 超参

| 项目 | 取值 |
|---|---|
| Backbone | [[ConvNeXt]]-Tiny |
| Input res | 228 × 228 |
| Loss | per-predicate BCE |
| 阈值 | 0.5 |

---

## 实验

### 数据集（三个 [[Visual Rearrangement Task]] 域）

| 数据集 | 规模 | 特点 | 用途 |
|---|---|---|---|
| **BlocksWorld** | 5,000 转移, 18 动作 | 3 blocks × 3 regions, 合成图 | sanity check, 验证能否恢复经典 STRIPS |
| **DinnerTable (sim)** | 12,000 转移, 70 动作 | 5 物体 × 7 摆放位, PyBullet 渲染 | 测试中等规模 |
| **DinnerTable Real** | 3,000 转移, 64 动作 | 真实桌面 RGB 图 | 真实场景泛化测试 |

### 实现细节

- **Stage 1**: ResNet + projection head + FSQ, EMA $\mu = 0.996$，$\lambda_{\text{inv}}, \lambda_{\text{sep}}$ 域相关
- **Stage 2**: Google OR-Tools **CP-SAT**, 300 s wall-clock budget, 字典序四层目标
- **Stage 3**: [[ConvNeXt]]-Tiny, BCE loss, 228×228 输入
- **测试时规划器**: BFS / [[A* Search]]，作用于 $m$ 维谓词空间
- **硬件**: 论文未明确

### Baselines

| 基线 | 类型 | 核心机制 |
|---|---|---|
| **WM-Rollout** | latent rollout | 在 FSQ code 空间 [[Beam Search]] 推 $K$ 步 |
| **WM-BFS** | latent graph search | 在观察过的 $(z, a, z')$ 边图上 BFS |
| **[[Latent Space Roadmap]]** (LSR) | continuous latent + 图 | 连续 embedding + clustering + 有向边 |
| **[[LatPlan]]-AMA3** | neuro-symbolic | 端到端学 binary latent + neural transition |

### 主要结果

- **BlocksWorld**: STRIPS-WM 在所有 horizon 上 **100% 成功**, 学到 9 谓词的精确 STRIPS 模型，$\sum\xi = \sum\eta = 0$
- **DinnerTable Sim**: STRIPS-WM 100% 成功；baseline 在 horizon ≥ 5 后明显下跌
- **DinnerTable Real**: STRIPS-WM 100% 成功；所有 baseline 在 long horizon 显著退化（Figure 4 直观对比）
- **抽象纠错**: DinnerTable Real 上视觉过分割 111 → 谓词合并回 71（真实状态数），证明 Stage 2 起到符号纠错作用

### 形式保证

- **Lemma 1 (Well-formed Operators)**: 任何可行解都给出合法 STRIPS 算子（add/del 互斥、pre$^+$/pre$^-$ 互斥）
- **Theorem 10 (Zero-Slack Soundness & Completeness)**: 当 $\sum\xi = \sum\eta = 0$ 时，学出的 STRIPS 模型对学习图是 sound and complete w.r.t. trusted missing pairs

---

## 批判性思考

### 优点

1. **可证明性**: 整个 Stage 2 的优化是布尔约束满足 + 字典序，零松弛时有 sound/complete 保证，这在 neural-symbolic 文献里是稀缺的
2. **可解释性**: 学出的谓词向量、precondition、add/delete mask 都可以打印出来人读，能直接喂给 PDDL 规划器
3. **长 horizon 鲁棒**: 因为最终规划在离散符号空间，不存在 latent rollout 误差累积
4. **抽象自纠错**: DinnerTable Real 上 Stage 2 把过分割的视觉 code 合并回正确状态数，说明符号学习对感知噪声有滤波效果
5. **数据高效**: DinnerTable Real 只用 3,000 转移就拿到 100%

### 局限性

1. **确定性假设**: 假设环境严格 STRIPS（确定性、无隐状态），不能直接处理随机性、部分可观或连续效果
2. **依赖 Stage 1 质量**: 如果视觉编码器把不同任务状态混叠（aliasing），Stage 2 解不出有效算子；论文承认这是主要失败模式
3. **不是 lifted relational**: 学到的是**接地命题**（grounded propositional）算子，换一组新物体或新动作元数就要重训，组合泛化能力受限
4. **CP-SAT 可扩展性**: 300 s 预算 + 字典序目标，在百状态/百动作规模可行，但更大领域（PDDL benchmark 级别）能否求解未验证
5. **真实图像测试规模有限**: 只在桌面级单视角 RGB 上验证，对动态背景、遮挡、多视角的泛化未知
6. **代码未开源**（截至 arXiv v1）

### 潜在改进方向

1. **Lifted 关系算子**: 把 CP-SAT 扩展到 first-order STRIPS（带变量的算子 schema），实现物体数泛化
2. **概率扩展**: 引入 probabilistic STRIPS / PPDDL，让 $\xi, \eta$ 解释成转移概率
3. **闭环执行**: 现在是 open-loop plan-then-execute，可以做 [[Receding-Horizon Execution]] 重规划应对扰动
4. **多视角输入**: 用 [[Multi-View Fusion]] / [[3D Gaussian Splatting]] 提供视角不变性
5. **与 LLM 任务规划组合**: 用 LLM 提供 high-level subgoal，STRIPS-WM 做 low-level 符号 grounding

### 可复现性评估

- [x] 训练细节较完整（Table 2/3/4 给了超参）
- [x] 数据集合成可重建（BlocksWorld / DinnerTable sim 描述清楚）
- [ ] 代码未开源
- [ ] 预训练模型未发布
- [ ] DinnerTable Real 数据集是否公开未明

---

## 关联笔记

### 基于

- [[LatPlan]]: 最直接的 neuro-symbolic 前驱，本文 baseline LatPlan-AMA3 是其代表
- [[JEPA]]: Stage 1 的 student-teacher 自监督思路源于 JEPA / I-JEPA
- [[FSQ]]: 离散状态编码的核心，相比 VQ-VAE 更稳定无 codebook collapse
- [[STRIPS]]: 1971 年经典规划形式，本文是其神经化版本

### 对比

- [[Latent Space Roadmap]] (LSR): 同为视觉规划，但缺乏算子结构，长 horizon 退化
- [[Dreamer]]: 视频世界模型 baseline，pixel-level rollout 与 STRIPS-WM 截然不同的路线
- [[minWM]]: 同样讨论 minimal world model，但走 latent 路线

### 方法相关

- [[Image-Grounded Task Graph]]: Stage 1 的核心产物
- [[CP-SAT]]: Stage 2 的求解器
- [[ConvNeXt]]: Stage 3 backbone
- [[A* Search]] / [[BFS]]: 测试时规划器
- [[Propositional Predicate]]: 学习的目标抽象
- [[Trusted Missing Pair]]: 负证据机制
- [[Closed-World Assumption]]: 论文中局部 CWA 假设
- [[Mask Disjointness]]: Lemma 1 的关键约束
- [[Visual Aliasing]]: 主要失败模式
- [[Inverse Dynamics]]: Stage 1 辅助损失
- [[Predictive Separation Loss]]: 减少 latent nondeterminism

### 任务相关

- [[Visual Rearrangement Task]]: 评估场景
- [[Propositional Planning]]: 任务范式

---

## 速查卡片

> [!summary] STRIPS-WM
> - **核心**: 从图像转移数据**端到端学经典 [[STRIPS]] 规划模型**（谓词 + precondition + add/delete），用 [[CP-SAT]] 求解算子学习的布尔优化
> - **三段管线**: Image →(JEPA + FSQ)→ 抽象任务图 →(CP-SAT 字典序优化)→ STRIPS 算子 →(ConvNeXt-Tiny)→ 视觉谓词分类器
> - **结果**: 三个 visual rearrangement 域上 100% 成功；BlocksWorld 上**精确恢复**人类工程化的 9 谓词 STRIPS 模型
> - **理论**: Lemma 1 well-formed + Theorem 10 zero-slack 时 sound & complete
> - **代码**: 暂未开源（arXiv v1, 2026-06-05）

---

*笔记创建时间: 2026-06-09*
