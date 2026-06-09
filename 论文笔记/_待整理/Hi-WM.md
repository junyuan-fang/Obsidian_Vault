---
title: "Hi-WM: Human-in-the-World-Model for Scalable Robot Post-Training"
method_name: "Hi-WM"
authors: [Yaxuan Li, Zhongyi Zhou, Yefei Chen, Yanjiang Guo, Jiaming Liu, Shanghang Zhang, Jianyu Chen, Yichen Zhu]
year: 2026
venue: arXiv
tags: [world-model, robot-post-training, human-in-the-loop, imitation-learning, manipulation, policy-improvement]
zotero_collection: _待整理
image_source: online
arxiv_html: https://arxiv.org/html/2604.21741
created: 2026-04-24
---

# 论文笔记：Hi-WM: Human-in-the-World-Model for Scalable Robot Post-Training

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Current Robotics, Tsinghua University, Peking University, University of Toronto |
| 日期 | April 2026 |
| 项目主页 | https://hi-wm.github.io/ |
| 对比基线 | [[Diffusion Policy]], [[Pi0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21741) / [Project](https://hi-wm.github.io/) |

---

## 一句话总结

> 将人类纠正干预从物理机器人转移到学习的 [[World Model]] 中，通过状态缓存与轨迹分支机制实现可扩展的机器人策略后训练。

---

## 核心贡献

1. **Human-in-the-World-Model 范式**: 首次提出在学习的世界模型中进行人类纠正干预式后训练，消除了物理执行的硬件瓶颈
2. **状态缓存与轨迹分支机制**: 支持 rollback 到失败前状态并生成多条纠正轨迹分支，大幅提升数据采集效率和多样性
3. **强真实世界对齐性**: 世界模型内评估与真实世界成功率高度相关（$r = 0.953$），三个操作任务平均提升 37.9 个百分点

---

## 问题背景

### 要解决的问题
大规模 [[预训练]] 得到的通用机器人策略在具体部署场景中仍然存在失败：接触丰富的动力学、环境特定变化和长时序误差累积都会导致策略失败。后训练（post-training）是将预训练策略变为可靠任务控制器的关键步骤。

### 现有方法的局限
现有 [[Human-in-the-Loop]] 后训练流程完全依赖物理执行：每次纠正都需要真实机器人时间、场景搭建、手动重置和操作员监督。随着任务规模扩大，后训练的边际成本急剧增长。与此同时，[[Action-Conditioned World Model]] 主要用于想象（imagination）、合成数据生成和策略评估，尚未被用作交互式纠正工作空间。

### 本文的动机
最有价值的监督信号往往位于学习者自身的失败状态附近——不是通用的成功 rollout，而是短小的纠正续集。如果能在世界模型中复现这些失败状态并让人类直接在其中干预纠正，就能以极低成本获得高质量的后训练数据。

---

## 方法详解

### 模型架构

Hi-WM 系统由三个核心组件构成：

- **[[Action-Conditioned World Model]]**: Visual Encoder → [[Latent Dynamics Model]] → Visual Decoder
- **Robot Policy**: 接收观测 $o_t$，生成连续动作 $a_t \in \mathbb{R}^{14}$（双臂系统：每臂 6-DoF + 1-DoF 夹爪）
- **Hardware-Agnostic Intervention Interface**: 支持键盘、标定机械臂、VR 控制器等多种输入设备，通过统一动作映射器转换为标准连续动作表示

### 核心模块

#### 模块1: 交互式世界模型（Interactive World Model）

**设计动机**: 世界模型需要在失败状态附近保持动作忠实性，不能仅仅学习"乐观"预测

**关键训练策略**:

1. **高维动作条件化（High-Dimensional Action Conditioning）**: 处理完整 14 维连续动作空间，支持双臂操作的 [[End-Effector]] 控制
2. **细粒度动作判别（Fine-Grained Action Discrimination）**: 训练数据中包含失败案例和偏离任务的状态，防止世界模型产生过度乐观的预测，确保能区分导致成功与失败的动作
3. **边缘情况数据覆盖（Edge-Case Data Coverage）**: 加入工作空间边界轨迹和精确夹爪对齐案例，维持世界模型到真实世界的对齐

#### 模块2: 状态缓存与回滚机制（State Caching and Rollback）

**设计动机**: 利用世界模型的可重置特性，从单次失败中生成多样化的纠正数据

**具体实现**:
- 在交互过程中缓存中间状态（latent states）
- 当 rollout 进入失败区域时，回滚到失败前的缓存状态
- 从同一缓存状态出发，人类操作员可以生成多条不同的纠正轨迹分支
- 单次模拟失败可以分支为多个不同的成功未来（diverse alternative futures）

#### 模块3: 纠正轨迹采集流程（Corrective Trajectory Collection）

**后训练数据生产流水线**:
1. 预训练策略在世界模型中闭环 rollout
2. 人类操作员在 rollout 变得不稳定或即将失败时介入
3. 纠正动作直接在世界模型中执行
4. 系统缓存中间状态，支持回滚到更早的时间步
5. 纠正片段与原始演示合并用于后训练

---

## 关键公式

### 公式1: [[Pearson Correlation Coefficient|世界模型-真实世界对齐度]]

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

**含义**: 衡量世界模型内策略评估成功率 $x$ 与真实世界成功率 $y$ 之间的线性相关性

**符号说明**:
- $x_i$: 第 $i$ 个策略配置在世界模型中的成功率
- $y_i$: 对应的真实世界成功率
- $r = 0.953$: 表明世界模型评估与真实执行高度一致

### 公式2: [[Action Space|动作空间表示]]

$$
a_t \in \mathbb{R}^{14}, \quad a_t = [a_t^{\text{left}}, a_t^{\text{right}}], \quad a_t^{\text{arm}} \in \mathbb{R}^{7}
$$

**含义**: 双臂机器人的连续动作空间定义，每臂 7 维（6-DoF 末端执行器位姿 + 1-DoF 夹爪开合）

**符号说明**:
- $a_t^{\text{left}}, a_t^{\text{right}}$: 左臂和右臂的动作向量
- $a_t^{\text{arm}} = [\Delta p_x, \Delta p_y, \Delta p_z, \Delta r_x, \Delta r_y, \Delta r_z, g]$: 位移增量 + 旋转增量 + 夹爪状态

### 公式3: [[Image Quality Metrics|图像质量评估指标]]

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right), \quad \text{SSIM} \in [0, 1], \quad \text{LPIPS} \in [0, 1]
$$

**含义**: 用于衡量世界模型预测图像与真实图像之间的视觉保真度

**符号说明**:
- $\text{PSNR}$ (Peak Signal-to-Noise Ratio): 越高越好，衡量像素级重建质量
- $\text{SSIM}$ (Structural Similarity Index): 越高越好，衡量结构相似性
- $\text{LPIPS}$ (Learned Perceptual Image Patch Similarity): 越低越好，衡量感知相似性

### 公式4: [[Cost Savings|成本节约计算]]

$$
\Delta C = C_{\text{real}} - C_{\text{Hi-WM}}
$$

**含义**: Hi-WM 框架相比真实世界后训练的累积成本节约，随场景覆盖范围扩大呈持续增长趋势

**符号说明**:
- $C_{\text{real}}$: 在真实世界中进行人类干预后训练的累积成本
- $C_{\text{Hi-WM}}$: 在世界模型中进行人类干预后训练的累积成本
- 在最大测试规模下节约约 $100,000 美元

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2604.21741v1/x1.png)

**说明**: Hi-WM 总览。策略在学习的 [[World Model]] 中 rollout，当 rollout 趋向失败时，人类提供短暂的纠正动作。缓存状态支持回滚和分支，为后训练生成多条续集轨迹。

### Figure 2: Pipeline / 完整流水线

![Figure 2](https://arxiv.org/html/2604.21741v1/x2.png)

**说明**: Hi-WM 完整流水线：(a) 预训练阶段从真实世界数据学习策略网络；(b) 策略在世界模型中闭环 rollout；(c) 当策略表现不佳时，人类通过硬件无关的输入设备（机械臂、键盘或 VR 控制器）介入，纠正动作直接在世界模型中执行；(d) 采集的干预片段加入训练集用于后训练。

### Figure 3: State Caching and Rollback / 状态缓存与回滚

![Figure 3](https://arxiv.org/html/2604.21741v1/x3.png)

**说明**: 状态缓存与回滚机制。Hi-WM 支持状态缓存和回滚，允许 rollout 回退到更早的状态并复用，从单个失败状态采集多条纠正分支，提升后训练数据的效率和多样性。

### Figure 4: Evaluated Tasks / 评估任务

![Figure 4](https://arxiv.org/html/2604.21741v1/x4.png)

**说明**: 三个代表性操作任务：Fold Towel（折叠毛巾，可变形物体协调）、Push-T（推 T 形物体，刚性物体位姿操作）、Route Rope（穿绳，柔性物体操作）。涵盖刚性和可变形物体交互。

### Figure 5: Hardware Setup / 硬件配置

![Figure 5](https://arxiv.org/html/2604.21741v1/x5.png)

**说明**: 左：遥操作系统由主从臂和顶视 RealSense D405 相机组成。右：真实世界操作精度对比——加入边缘情况数据增强后，世界模型目标位置与物理执行的定位偏差大幅减小。

### Figure 6: Correlation, Cost, and Scaling / 对齐性、成本与扩展性

![Figure 6](https://arxiv.org/html/2604.21741v1/x6.png)

**说明**: (a) 世界模型与真实世界成功率的 [[Pearson Correlation Coefficient|Pearson 相关系数]] 达 $r = 0.953$，表明世界模型评估与真实执行高度一致；(b) 在世界模型中进行人类干预相比真实世界的累积成本节约，随场景覆盖扩大优势持续增长；(c) 随着干预数据增加，真实世界成功率在三个任务上单调提升。

### Figure 7: Generalization / 泛化性能

![Figure 7](https://arxiv.org/html/2604.21741v1/x7.png)

**说明**: 后训练策略在 Push-T 任务上的真实世界泛化提升。左：三种泛化设置可视化（外观变化、背景变化、干扰物变化）。右：定量对比显示，使用 Hi-WM 采集的纠正数据后训练的策略在所有泛化场景下均大幅优于基准策略。

### Table 1: 边缘情况数据对视觉保真度的影响

| 比例 (%) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----------|--------|--------|---------|
| 0 (Base) | 18.50 | 0.815 | 0.152 |
| 20 | 19.82 | 0.853 | 0.124 |
| 50 | 21.15 | 0.901 | 0.098 |
| 100 (Full) | 22.53 | 0.942 | 0.055 |

**表格说明**: 随着边缘情况轨迹占比增加，世界模型的图像预测质量在三个指标上全面提升。完整边缘情况数据使 PSNR 提升 4.03 dB，LPIPS 降低 63.8%。

### Table 2: 标准任务设置下的真实世界成功率 (%)

| 方法 | 策略 | Fold Towel | Push-T | Route Rope |
|------|------|-----------|--------|-----------|
| Base | DP | 42.1 | 52.9 | 47.0 |
| Base | π₀ | 55.3 | 76.5 | 64.7 |
| WM-CL | DP | 76.3 | 64.7 | 70.6 |
| WM-CL | π₀ | 78.9 | 79.4 | 82.4 |
| **Hi-WM** | **DP** | **92.1** | **85.3** | **94.1** |
| **Hi-WM** | **π₀** | **97.4** | **97.1** | **100.0** |

**表格说明**: Hi-WM 在所有 6 个策略-任务组合中均取得最佳结果。相比基准策略平均提升 37.9 个百分点，相比 WM-CL 基线平均提升 19.0 个百分点。π₀ + Hi-WM 在 Route Rope 上达到 100% 成功率。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实世界演示数据 | 未明确 | 双臂操作三类任务 | 预训练 |
| 边缘情况轨迹 | 约 30% 常规数据量 | 工作空间边界、精确对齐 | 世界模型增强 |
| Hi-WM 纠正数据 | 迭代采集（20%→35%） | 失败状态附近的纠正轨迹 | 后训练 |

### 实现细节

- **机器人平台**: 双臂 YAM-Ultra 平台
- **动作维度**: 14D 连续动作（每臂 6-DoF + 1-DoF 夹爪）
- **观测**: 顶视 RealSense D405 相机图像 + 双臂初始位姿
- **执行频率**: 15 Hz
- **策略骨干**: [[Diffusion Policy]] (DP) 和 [[Pi0|π₀]]
- **世界模型**: Visual Encoder + [[Latent Dynamics Model]] + Visual Decoder

### 可视化结果

- 世界模型内 10 分钟连续 Push-T 交互稳定运行，证明模型支持工作空间内的插值而非稀疏锚点记忆
- 边缘情况数据增强后，重复定位精度在工作空间边界区域显著提升
- 后训练策略在外观/背景/干扰物三种泛化设置下均优于基准策略

---

## 批判性思考

### 优点
1. **范式创新**: 首次将 [[Human-in-the-Loop]] 纠正从物理世界转移到世界模型中，开辟了全新的后训练范式
2. **数据效率**: 状态缓存与轨迹分支机制使单次失败产生多条纠正轨迹，极大提升数据效率
3. **经济性**: 累积成本节约可达约 10 万美元，且随规模扩大优势持续增长
4. **强对齐性**: $r = 0.953$ 的高 Pearson 相关系数证明世界模型评估可靠替代真实世界评估

### 局限性
1. **世界模型质量依赖**: 框架效果完全取决于世界模型的保真度，对于高度动态或接触丰富的场景（如灵巧手操作），当前世界模型可能不够准确
2. **任务多样性有限**: 仅在三个桌面操作任务上验证，未涉及移动操作、多步长时序任务或室外环境
3. **边缘情况数据仍需真实采集**: 增强世界模型所需的边缘情况轨迹仍然来自真实世界，这部分成本未被消除
4. **泛化评估范围窄**: 仅在 Push-T 一个任务上测试泛化，且变化类型较为简单

### 潜在改进方向
1. 将框架扩展到更高维度的操作任务（如灵巧手、全身操作）
2. 结合 [[Diffusion Model]] 等生成模型提升世界模型的视觉保真度和动力学准确性
3. 自动化失败检测，减少人类操作员判断介入时机的负担
4. 探索世界模型自身的在线更新机制，使其随纠正数据一起持续改进

### 可复现性评估
- [ ] 代码开源（项目主页已发布但代码链接待确认）
- [ ] 预训练模型（未明确提供）
- [x] 训练细节完整（动作维度、执行频率等）
- [ ] 数据集可获取（未公开）

---

## 关联笔记

### 基于
- [[Diffusion Policy]]: 作为策略骨干之一（DP）
- [[Pi0]]: 作为策略骨干之一（π₀），来自 Physical Intelligence

### 对比
- [[World Model]]: Hi-WM 将世界模型从被动评估工具变为主动纠正工作空间
- [[DAgger]]: 传统 human-in-the-loop 模仿学习方法，依赖物理执行

### 方法相关
- [[Action-Conditioned World Model]]: 核心组件，支持高维动作条件化
- [[Latent Dynamics Model]]: 世界模型的潜在动力学模块
- [[Human-in-the-Loop]]: 后训练范式的理论基础
- [[Imitation Learning]]: 预训练和后训练均基于模仿学习框架

### 硬件/数据相关
- [[YAM-Ultra]]: 双臂机器人平台
- [[RealSense D405]]: 顶视深度相机

---

## 速查卡片

> [!summary] Hi-WM: Human-in-the-World-Model for Scalable Robot Post-Training
> - **核心**: 在学习的世界模型中进行人类纠正干预式后训练
> - **方法**: 状态缓存 + 轨迹分支 + 硬件无关干预接口
> - **结果**: 平均提升 37.9pp，世界模型-真实世界相关性 r=0.953
> - **代码**: https://hi-wm.github.io/

---

*笔记创建时间: 2026-04-24*
