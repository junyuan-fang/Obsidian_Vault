---
title: "PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation"
method_name: "PiL-World"
authors: [Chong Ma, Taiyi Su, Jian Zhu, Jianjun Zhang, Zitai Huang, Yi Xu, Hanli Wang]
year: 2026
venue: arXiv
tags: [world-model, vla-evaluation, closed-loop-rollout, action-conditioned-video, dual-arm-manipulation, video-diffusion, policy-in-the-loop]
zotero_collection: 1-世界模型与视频生成
image_source: online
arxiv_html: https://arxiv.org/html/2606.05773v1
created: 2026-06-08
---

# 论文笔记：PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tongji University (推断自作者群) |
| 日期 | June 2026 |
| 项目主页 | https://pil-world.github.io |
| 对比基线 | [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05773) / [HTML](https://arxiv.org/html/2606.05773v1) / [Code](https://pil-world.github.io) |

---

## 一句话总结

> PiL-World 是首个**显式匹配 VLA 闭环接口**的世界模型：以 chunk 为粒度，与 VLA 推理交替执行，把"想象 rollout 的成功率误差"从 63.2% 压到 12.0%。

---

## 核心贡献

1. **Chunk-Wise 闭环 World Model**: 把世界模型预测对齐到 VLA 的 action chunk 粒度，每轮生成 $K$ 帧多视角未来观测后**反馈给 VLA 重新查询**，复现真机闭环的"观测—决策—执行"循环，无需真机每步介入。
2. **Action-Derived Visual Control**: 通过正运动学 + 相机标定，把绝对关节命令投影成 head-view 上的 gripper 标记图像帧，再用 latent history memory 维持跨轮上下文，让动作条件更"视觉化"且时序一致。
3. **联合多视角未来预测**: 在 latent 空间同时去噪 top / left-wrist / right-wrist 三视角的未来帧，保证视图间几何与物理一致。
4. **显式包含失败轨迹的微调**: fine-tune 数据混入遥操作失败轨迹，让想象 rollout 的"成败分布"与真机执行更对齐，避免世界模型"过度乐观"地把所有 rollout 都拉向成功态。
5. **三项真机双臂任务验证**: 在 Sort Cubes / Stack Bowls / Stack Blocks 上，相对 [[Ctrl-World]] 把 $|\Delta SR|$ 从 63.2% 降到 **12.0%**，HFR 从 41.5% 提到 **70.1%**，task-checkpoint 维度 Pearson 相关达到 **0.94**。

---

## 问题背景

### 要解决的问题

真机评估 [[VLA]] 策略既慢又贵，更换 checkpoint 就要重复跑数百次任务。理想做法是用一个 **[[World Model|世界模型]]** 在 silico 跑 rollout 当 proxy，但 VLA 在真机里运行的是**闭环 chunk-wise** 协议：观察 $\mathbf{x}_t$ → 预测 chunk $A_t$ → 执行 → 重新观察 → 重新查询策略。世界模型必须**匹配这套闭环接口**，imagined success rate 才能近似真机 success rate。

### 现有方法的局限

- **Open-loop video prediction**（WorldEval / Hi-WM / 大多数 [[Action-Conditioned Video Diffusion]] 工作）：沿着**预先采集好的整段 action 轨迹**生成视频，不存在"生成观测反馈给策略"的环节，所以测的是"动作序列的可视化保真度"而不是"策略的成败率"。
- **现有 closed-loop 世界模型**（如 [[Ctrl-World]]）：粒度粗、跨轮上下文丢失、训练只见成功轨迹，导致 imagined rollout 全都"乐观地"走向成功，与真机失败率脱节——本文测得其 $|\Delta SR|$ 高达 71.8%（Sort Cubes）。
- **动作条件耦合方式弱**：大多直接把 7/14 维关节向量 cross-attend 到 video diffusion 上，缺少**视觉化的、与画面像素对齐的动作信号**，导致末端运动幅度与画面不一致。

### 本文的动机

作者认为，要让 imagined rollout 真正可用，必须**一对一复刻 VLA 真机执行栈的接口**：

1. **同样的 chunk 粒度**：策略一次出 $H_\pi$ 步动作，世界模型就要一次出 $K \cdot \Delta$ 帧未来观测，把终点帧交回策略。
2. **同样的多视角输入**：策略训练用 top + 双 wrist 三视角，世界模型也要联合预测三视角。
3. **同样的成败分布**：真机会失败，世界模型不能只见过成功示范；要把失败遥操作轨迹塞进去。
4. **同样的视觉动作语义**：把动作"画"在 head-view 里当 visual condition，比纯数值条件更鲁棒。

---

## 方法详解

### 模型架构

PiL-World 采用 **chunk-wise latent video diffusion** 架构，与 [[VLA]] 策略 $\pi$ 在外层交替循环：

- **输入（每轮 r）**:
  - 当前多视角观测 $\mathbf{x}_t = \{x_t^v\}_{v \in \mathcal{V}}$，$\mathcal{V}=\{\text{top, left-wrist, right-wrist}\}$
  - latent history memory $\mathcal{H}_t$（最近 $H_h{=}5$ 帧的 VAE latent）
  - 由 VLA 预测的 action chunk 经 $\Gamma(\cdot)$ 投影出的 head-view 视觉控制帧序列 $C_t$
  - 任务指令 $g$（自然语言）
- **Backbone**: [[Wan2.1]]-14B（视频扩散基座，LoRA 微调）
- **核心模块**:
  - [[Action-to-Control Projection]]：把关节命令 → gripper 像素标记
  - [[Latent History Memory]]：跨轮维护视觉上下文
  - 联合多视角去噪：在 latent 维度拼接 $\{Z_t^{v,f}\}_{v\in\mathcal{V}}$ 同步生成
- **输出**: stride 对齐的多视角未来观测 $\hat{\mathbf{x}}_{t+\Delta:t+K\Delta:\Delta}$，共 $K{=}15$ 帧
- **闭环驱动**: 终点帧 $\hat{\mathbf{x}}_{t+K\Delta}$ 反馈给 $\pi$ 作为下一轮观测，串成 $R{=}5$ 轮、覆盖 $R\cdot K\cdot\Delta{=}225$ 个原始时间步的稠密想象轨迹。

### 核心模块

#### 模块1: Chunk-Wise Closed-Loop Rollout Pipeline

**设计动机**: 真机里 VLA 一次出 $H_\pi{=}50$ 步动作，机器人执行一段后才会重新观察、再次查询；世界模型也必须按"块"产生观测、再交回策略，才能与真机闭环接口对齐。

**具体实现**:
- 每轮 $r$ 执行 4 步：①$\pi$ 预测 $A_t$ → ②$\Gamma(A_t^{\Delta,K})$ 投影为 visual control → ③$W_\theta$ 联合生成 $K$ 帧多视角未来 → ④终点帧 + 更新后的 proprioceptive state 输入下一轮。
- 相邻轮共享 latent history memory，避免长程漂移。
- stride $\Delta{=}3$ 把 50 步原始动作降采样到 15 个控制锚点，平衡时间分辨率与生成成本。

#### 模块2: [[Action-to-Control Projection]] $\Gamma(\cdot)$

**设计动机**: 用纯数值条件指挥扩散模型很难精确控制末端位置；把动作"画"成 head-view 上的视觉标记，让扩散过程获得**像素对齐的几何先验**。

**具体实现**:
- 用机器人[[Inverse Kinematics|正/逆运动学]]把 14 维关节命令转成左右 gripper 的世界坐标。
- 用 head 相机内外参把 gripper 位置投影到 head-view 图像平面，画成圆形 marker。
- **Marker 大小编码 gripper 开闭状态**——闭合越紧 marker 越小，实现连续可微的状态编码。
- 输出 stride-aligned 视觉控制序列 $C_t = \{c_{t+\Delta},\dots,c_{t+K\Delta}\}$，长度 $K$ 与生成帧对齐。
- 这是**确定性几何投影**，没有可学习参数，因此不会引入额外训练复杂度。

#### 模块3: [[Latent History Memory]] $\mathcal{H}_t$

**设计动机**: 单步条件不足以保证跨轮（cross-round）的物体位置 / 光照 / 手臂姿态一致，必须把最近若干帧的"语义记忆"显式带进生成。

**具体实现**:
- VAE 编码器 $E_\phi$（来自 [[Wan2.1]] 基座）把每个视角的最近 $H_h$ 帧压成 latent token：$Z_t^{v,h} = E_\phi(\{x_\tau^v\}_{\tau\in\mathcal{I}_t^h})$
- 当前帧 latent $Z_t^{v,0}$ + 历史 latent $Z_t^{v,h}$ + 视觉控制 $C_t$ + 指令 $g$ 一起作为条件，只对未来 latent $Z_t^{v,f}$ 去噪。
- 跨轮时滑窗更新 $\mathcal{I}_t^h$，把刚生成的帧并入历史。
- 消融实验表明，去掉这块 LPIPS 飙升 **3 倍以上**（见 Table A.2）。

#### 模块4: Success/Failure Fine-Tuning Data

**设计动机**: VLA 策略真机失败率不低（Stack Blocks 上 $\pi_0$ 真机 SR 仅 50%），若世界模型只见成功示范，imagined rollout 必然"过度乐观"，与真机 SR 拉开巨大 gap。

**具体实现**:
- **Pretrain**：在 RealSource World（自建大规模机器人视频库，14M 帧 / 11,428 episodes / 35 tasks）上学通用机器人–环境动力学。
- **Fine-tune**：每个目标任务收集（successful demonstrations + failed teleoperated executions）混合数据，让世界模型"亲眼见过"失败轨迹的物理后果——例如手臂错过物体、物体倒下、抓取打滑等。
- 用 LoRA 适配，保留基座生成能力的同时注入任务先验。

---

## 关键公式

### 公式1: [[Action Chunking|VLA 动作块预测]]

$$
A_t = \{a_{t+1}, a_{t+2}, \ldots, a_{t+H_\pi}\} = \pi(\mathbf{x}_t, \mathbf{s}_t, g)
$$

**含义**: VLA 策略 $\pi$ 一次性输出长度为 $H_\pi$ 的动作块（论文里 $H_\pi{=}50$ 维 14 的绝对关节命令），这是闭环的"动作侧锚点"。

**符号说明**:
- $\mathbf{x}_t$: 当前多视角观测
- $\mathbf{s}_t$: 机器人 proprioceptive 状态
- $g$: 自然语言任务指令
- $a_{t+i} \in \mathbb{R}^{14}$: 双臂绝对关节命令

### 公式2: Stride-Aligned 动作子序列

$$
A_t^{\Delta,K} = \{a_{t+\Delta}, a_{t+2\Delta}, \ldots, a_{t+K\Delta}\}
$$

**含义**: 把 $H_\pi$ 步原始动作按 stride $\Delta$ 降采样到 $K$ 个锚点，与世界模型一轮生成的 $K$ 帧观测一一对齐。

**符号说明**:
- $K{=}15$: 每轮预测帧数
- $\Delta{=}3$: stride
- $K\cdot\Delta{=}45 \le H_\pi{=}50$，留出余量

### 公式3: 多视角未来观测生成

$$
\hat{\mathbf{x}}_{t+\Delta:t+K\Delta:\Delta} \sim W_\theta(\cdot \mid \mathbf{x}_t, \mathcal{H}_t, \Gamma(A_t^{\Delta,K}), g)
$$

**含义**: 世界模型 $W_\theta$ 以当前观测 / 历史记忆 / 视觉控制 / 指令为条件，采样出未来 $K$ 帧多视角观测。

**符号说明**:
- $W_\theta$: 基于 [[Wan2.1]]-14B + LoRA 的多视角 latent video diffusion
- $\mathcal{H}_t$: latent history memory
- $\Gamma$: action-to-control 投影（确定性）

### 公式4: [[Action-to-Control Projection|动作—视觉控制投影]]

$$
C_t = \Gamma(A_t^{\Delta,K}) = \{c_{t+\Delta}, c_{t+2\Delta}, \ldots, c_{t+K\Delta}\}
$$

**含义**: 把每个动作锚点通过正运动学 + 相机投影变成一张 head-view 标记图，作为 latent 扩散的视觉条件。

**符号说明**:
- $c_{t+k\Delta}$: 第 $k$ 个锚点对应的 head-view 控制帧，包含左右 gripper 位置 marker，marker 大小编码开闭。

### 公式5: [[Latent History Memory|历史 latent 编码]]

$$
Z_t^{v,h} = E_\phi(\{x_\tau^v\}_{\tau \in \mathcal{I}_t^h})
$$

**含义**: 把每个视角 $v$ 在历史索引集 $\mathcal{I}_t^h$ 上的帧用 VAE 编码器 $E_\phi$ 压成 latent，存入 $\mathcal{H}_t = \{Z_t^{v,h}\}_{v\in\mathcal{V}}$。

**符号说明**:
- $E_\phi$: [[Wan2.1]] 视频 VAE 编码器
- $\mathcal{I}_t^h$: 当前时刻往前数 $H_h{=}5$ 帧的索引集合

### 公式6: 当前帧 latent

$$
Z_t^{v,0} = E_\phi(x_t^v)
$$

**含义**: 与历史 latent 同一编码空间下的"现在锚点"，避免训练 / 推理的编码不一致。

### 公式7: 未来目标 latent

$$
Z_t^{v,f} = E_\phi(\{x_\tau^v\}_{\tau=t+\Delta:t+K\Delta:\Delta})
$$

**含义**: 训练时 stride 采样未来真实帧编成 latent，去噪目标只作用在 $Z_t^f = \{Z_t^{v,f}\}_{v\in\mathcal{V}}$ 上，**当前帧 latent 与历史 latent 不参与去噪**——只做条件。这是 PiL-World 与普通 video diffusion 的关键区别。

### 公式8: Real–Imagined Success Agreement

$$
|\Delta SR| = |SR_{\text{imag}} - SR_{\text{real}}|
$$

**含义**: 想象成功率与真机成功率的绝对差，是评测世界模型"是否可信"的核心指标，越低越好。

### 公式9: [[Hallucination-Free Ratio|HFR]]

$$
HFR = \frac{1}{M}\sum_{m=1}^M \frac{1}{N}\sum_{i=1}^N \frac{t_h^{(i,m)}}{T_i}
$$

**含义**: 想象 rollout 在出现"显著幻觉"前可信的归一化长度，刻画"想象多久之后就会崩"。

**符号说明**:
- $M$: 标注员数
- $N$: 评估的 rollout 数
- $t_h^{(i,m)}$: 第 $m$ 个标注员在第 $i$ 条 rollout 上标的"首次明显幻觉"帧
- $T_i$: 该 rollout 总长

### 公式10: Pearson 相关度（task-checkpoint 维度）

PiL-World 的 $SR_{\text{imag}}$ 与 $SR_{\text{real}}$ 跨 (task, checkpoint) 的 Pearson 相关 $\rho{=}0.94$，意味着即使绝对值有偏，"哪个 checkpoint 更强"的排序仍高度一致——这对 VLA 的**模型选择 / early-stopping** 是黄金属性。

---

## 关键图表

### Figure 1: Motivation — 闭环 vs 开环 rollout

![Figure 1](https://arxiv.org/html/2606.05773v1/Figures/Figure1.png)

**说明**: (a) Policy-in-the-Loop 协议：世界模型预测未来观测，**反馈给 VLA 重新查询**；(b) 与 open-loop 对比：开环沿预定动作走，不重新查策略；(c) 强调"想象 rollout 必须与真机执行轨迹保持一致"才能当评估 proxy。这张图直接定义了 PiL-World 与上一代世界模型评估工作的接口差异。

### Figure 2: PiL-World 总览

![Figure 2](https://arxiv.org/html/2606.05773v1/Figures/Figure2.jpg)

**说明**: 四象限拆解整套系统——(a) chunk-wise 闭环 rollout pipeline；(b) [[Action-to-Control Projection]] 把 14 维关节投影到 head-view marker；(c) success/failure 混合微调数据；(d) [[Latent History Memory]] 维持跨轮上下文。这是论文的"一图懂"图。

### Figure 3: Real–Imagined Success Agreement

![Figure 3](https://arxiv.org/html/2606.05773v1/Figures/rollout_success_trend.png)

**说明**: 散点 = (task, VLA checkpoint) 组合，形状区分任务、颜色区分 checkpoint。横轴 $SR_{\text{real}}$、纵轴 $SR_{\text{imag}}$。PiL-World 点云沿对角线分布（$\rho{=}0.94$），证明它能保留"哪个 checkpoint 更好"的真实排序。

### Figure 4: Action-Horizon LPIPS Gain

![Figure 4](https://arxiv.org/html/2606.05773v1/Figures/lpips_gain_pilworld_square_legend_v3.png)

**说明**: 横轴=预测帧序号，纵轴=Ctrl-World LPIPS − PiL-World LPIPS。**随时间正向 gain 逐步放大**，说明 PiL-World 在长时间预测上更稳——这正是闭环 rollout 最需要的性质。

### Figure 5(a): Sort Cubes Qualitative Rollout

![Figure 5a](https://arxiv.org/html/2606.05773v1/Figures/sortcubes_example.png)

**说明**: 真机执行 vs 想象 rollout 三视角逐帧对比。PiL-World 的想象帧在抓取、移动、堆叠等关键节点都与真机几乎吻合，对比 Ctrl-World 的累计漂移肉眼可见。

### Figure 5(b): Stack Bowls Qualitative Rollout

![Figure 5b](https://arxiv.org/html/2606.05773v1/Figures/stackbowls_example.png)

**说明**: 第二个任务的视觉对比，重点看"碗叠到碗里"这种 contact-rich 时刻 PiL-World 是否能保住几何一致——结果是基本能。

### Figure A.1: 三项任务示例与子任务分割

![Figure A.1](https://arxiv.org/html/2606.05773v1/Figures/task_introduce.png)

**说明**: Sort Cubes / Stack Bowls / Stack Blocks 各自的真实台面图与每个子任务的关键帧标注，解释为什么 60 / 54 / 40 个子任务可以作为细粒度评测单元。

### Figure A.2: Action-to-Control Projection Pipeline

![Figure A.2](https://arxiv.org/html/2606.05773v1/Figures/action_projection.png)

**说明**: 详细拆解 $\Gamma$ 的三步——正运动学算 gripper 世界坐标 → head-view 投影 → marker 绘制（大小编码开闭）。明确这是**纯几何过程，无学习参数**。

### Figure A.3: RealSource World 单步预测定性结果

![Figure A.3](https://arxiv.org/html/2606.05773v1/Figures/realsource_qualitative_grid.png)

**说明**: 预训练阶段的多视角单步预测样例，证明 RealSource World 的规模和多样性足以支撑通用机器人–环境动力学先验。

### Table 1: Closed-Loop Rollout Agreement（核心结果）

| Task | $SR_{\text{real}}$ | Ctrl-World $SR_{\text{imag}}$ | Ctrl-World $\|\Delta SR\|$ | PiL-World $SR_{\text{imag}}$ | PiL-World $\|\Delta SR\|$ | Ctrl-World HFR | PiL-World HFR |
|------|---|---|---|---|---|---|---|
| Sort Cubes  | 83.3% | 11.5% | 71.8% | 68.3% | **15.0%** | 39.5% | **83.3%** |
| Stack Bowls | 96.7% | 24.1% | 72.6% | 92.5% | **4.2%**  | 47.4% | **83.9%** |
| Stack Blocks| 50.0% | 4.9%  | 45.1% | 33.3% | **16.7%** | 37.7% | **43.0%** |
| **Average** | —     | —     | 63.2% | —     | **12.0%** | 41.5% | **70.1%** |

**说明**: 主表。三任务平均 $|\Delta SR|$ 从 63.2% → **12.0%**（−51.2 pt），HFR 从 41.5% → **70.1%**（+28.6 pt）。Stack Blocks 上 gap 仍偏大，因为这是 contact-rich 程度最高的任务，小误差会被堆叠几何放大。

### Table 2: Single-Step Multi-View Prediction LPIPS

| Task | Method | Overall | Head | Wrist Avg |
|------|--------|---------|------|-----------|
| Sort Cubes  | Ctrl-World | 0.1454 | 0.1030 | 0.1666 |
| Sort Cubes  | **PiL-World** | **0.0965** | **0.0597** | **0.1148** |
| Stack Bowls | Ctrl-World | 0.1366 | 0.0959 | 0.1569 |
| Stack Bowls | **PiL-World** | **0.1100** | **0.0597** | **0.1351** |
| Stack Blocks| Ctrl-World | 0.1277 | 0.0885 | 0.1474 |
| Stack Blocks| **PiL-World** | **0.1208** | **0.0617** | 0.1503 |

**说明**: 相对降幅 Sort Cubes −33.7% / Stack Bowls −19.5% / Stack Blocks −5.4%。head-view 上 PiL-World 优势最显著（−42% ~ −38%），这与 head-view 是 visual control 直接作用对象一致；wrist 视角是"被动"传播过去的，提升相对小。

### Table A.2: Latent History Memory 消融

| Task | With Memory (LPIPS) | Without Memory (LPIPS) | 相对恶化 |
|------|---|---|---|
| Sort Cubes   | 0.0965 | 0.3146 | **3.26×** |
| Stack Bowls  | 0.1100 | 0.2759 | **2.51×** |
| Stack Blocks | 0.1208 | 0.3403 | **2.82×** |

**关键发现**: 去掉 history memory 后 LPIPS 普遍恶化 2.5–3.3 倍，说明跨轮 latent 上下文是 PiL-World 闭环一致性的**最关键支柱**，比单纯增大 LoRA rank 或训练步数都更重要。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RealSource World（自建） | 14M+ 帧 / 11,428 episodes / 35 tasks | 多任务多机器人通用机器人–环境视频库 | 预训练 |
| Sort Cubes | 100 train / 20 test episodes, 60 test subtasks, 8707/1830 clips | 双臂，按大小排序 3 个方块 | 微调 + 评测 |
| Stack Bowls | 100 / 18 episodes, 54 subtasks, 7342/1301 clips | 双臂，3 个碗叠塔 | 微调 + 评测 |
| Stack Blocks | 200 / 20 episodes, 40 subtasks, 15626/820 clips | 双臂，3 个积木叠塔（contact-rich 最强） | 微调 + 评测 |

每个任务的微调数据都**显式混合成功示范 + 失败遥操作**，量级未在主表披露具体比例。

### 实现细节

- **Backbone**: [[Wan2.1]]-14B 视频扩散基座
- **适配**: LoRA 微调（rank 未具体披露）
- **VLA 策略**: $\pi_0$（仅在成功示范上微调，PiL-World 评测时**冻结策略**）
- **预测窗**: $K{=}15$ 帧 / 轮，stride $\Delta{=}3$，每轮覆盖 $K\Delta{=}45$ 原始步
- **滑窗**: stride-sampled 9 帧滑窗
- **闭环轮数**: 最多 $R{=}5$ 轮 / 想象轨迹 ⇒ 覆盖 225 个原始时间步
- **VLA action chunk**: $H_\pi{=}50$，每步 14 维绝对关节命令
- **历史帧数**: $H_h{=}5$
- **图像分辨率**: $224\times224$
- **视角**: $\mathcal{V}=\{$top, left-wrist, right-wrist$\}$，三视角联合预测
- **优化**:
  - 预训练 22 epochs，64× **H20** GPU
  - 微调 20 epochs，8× H20 GPU
- **基线**: [[Ctrl-World]]（当前 SOTA policy-compatible world model），horizon-matched 协议对齐

### 评测协议

- **闭环对齐**: imagined 分支与 real 分支从**同一子任务起始观测 + 同一指令 + 同一冻结 VLA**出发，rollout 完后比对成败。
- **指标三件套**: $|\Delta SR|$（绝对差）、Pearson $\rho$（排序保真）、HFR（rollout 可信长度）、单步 LPIPS（视觉保真）。
- HFR 由人工标注首次明显幻觉帧。

### 可视化结果

- Sort Cubes / Stack Bowls 上 PiL-World 的想象 rollout 在抓取节点、移动轨迹、堆叠对齐上都与真机高度一致；Ctrl-World 在第 2 轮起明显漂移、物体"凭空挪动"。
- LPIPS gain 曲线（Figure 4）表明优势在第 5–15 帧持续扩大，意味着越长 rollout 越能体现 PiL-World 的价值。
- Wrist 视角 LPIPS 改善幅度小，是因为 visual control 只画在 head-view 上，wrist 视图靠 latent 传播——这是后续改进方向之一。

---

## 批判性思考

### 优点

1. **接口对齐很硬核**：直接复刻 VLA 真机闭环协议（chunk 粒度 + 多视角 + 反馈循环），从评测语义上根本性地胜过 open-loop video prediction 类工作。
2. **Action-derived visual control 设计精妙**：把数值动作"画进画面"，让扩散模型获得像素对齐的几何先验，比 cross-attention 关节向量条件更直观也更鲁棒；且 $\Gamma$ 是确定性几何过程，无额外参数，几乎是"白送的"先验注入。
3. **Success/Failure 数据对齐**：很多 World Model 工作只学成功示范，本文显式纳入失败遥操作，这是**让 imagined SR 不再过度乐观**的关键 ingredient，简单但非常对症。
4. **指标设计有想法**：除常规 $|\Delta SR|$ 外还引入 HFR 与 Pearson $\rho$，前者刻画"可信长度"，后者刻画"排序保真"——对实际 model selection 用例极友好。
5. **消融做得扎实**：Table A.2 直接证明 latent history memory 是核心，2.5–3.3× LPIPS 的差距让设计动机不再是"经验之谈"。

### 局限性

1. **任务面太窄**：只在 3 个双臂桌面任务上验证，且都是 stacking / sorting 类静态目标任务，对移动操作 / 长时序 / 软体物体 / 非视觉富集任务都未触及。
2. **Contact-rich 任务仍有差距**：Stack Blocks 上 $|\Delta SR|{=}16.7\%$、HFR 仅 43%，说明扩散模型对接触动力学的物理一致性仍是短板（这是整个 video world model 圈共同痛点）。
3. **HFR 人工依赖**：HFR 需多名标注员标注首次幻觉帧，规模化代价高，难以做大量 ablation 或 cross-team reproducibility。
4. **Head-view 中心化条件**：$\Gamma$ 只在 head-view 上画 marker，当 head 视角遮挡或任务以 wrist 视角为主时（如灵巧手内手指接触），条件信号会失效。
5. **基线略单**：主表只对 Ctrl-World，缺与 WorldVLA / dWorldEval / WorldGym 等同期方法的横向比较，难判断 PiL-World 之于"接口闭环"的增益与"训练数据规模"的增益各占多少。
6. **未做 VLA 多样性**：只用 $\pi_0$ 一个策略，PiL-World 对其他 VLA（OpenVLA / RDT / GR-2 / Pi 0.5 ...）的迁移性未验证；不同策略的 action chunk 长度 / 视角设定差异都可能影响表现。
7. **计算成本不轻**：14B 基座 + 多视角联合扩散，每轮要去噪三视角 15 帧，闭环 5 轮 = 一次想象 rollout 要去噪 225 帧 latent，与真机评测相比省下的是物理时间，付出的是 GPU 时间。

### 潜在改进方向

1. **多视角 visual control**：把 $\Gamma$ 同时画在左右 wrist 视角，wrist 视角 LPIPS 应有显著提升；可联合 Bayesian fusion 决定哪个视角作为主控制源。
2. **物理 prior 注入**：在 contact-rich 帧引入碰撞 / 刚体先验（如 differentiable physics 或 SDF 约束），缓解 Stack Blocks 漂移。
3. **自动化 HFR**：用 VQA 模型或者 VLA-as-judge 自动检测"幻觉首帧"，把 HFR 从评测瓶颈变成可大规模采集的训练信号。
4. **更长 horizon 闭环**：现在 $R{=}5$ 轮 / 225 步；若能配合 [[Causal Consistency Distillation]] 把单步去噪压到 4 step，闭环长度可拓到 10–20 轮，覆盖整条 episode。
5. **失败轨迹自动合成**：用扰动 / 噪声注入自动生成"伪失败"轨迹，减少遥操作采集成本。
6. **跨 VLA 评测协议化**：把"接口对齐"做成模块化 spec，让任何 chunk-wise VLA 都可即插即用做想象评估。

### 可复现性评估

- [x] 项目页发布
- [ ] 代码开源（项目页未明确，需核对最终版本）
- [ ] 预训练模型（RealSource World pretrain 权重 14B，开源概率较低）
- [x] 训练细节相对完整（epoch / GPU / 超参均披露）
- [ ] 数据集可获取（RealSource World 是自建，三项目标任务数据规模披露但是否公开未明）
- [x] 指标定义明确（$|\Delta SR|$ / HFR / LPIPS 均给公式）

---

## 关联笔记

### 基于

- [[Wan2.1]]: 视频扩散基座
- [[Action-Conditioned Video Diffusion]]: 大方向先驱
- [[Action Chunking]]: VLA 输出粒度
- [[VLA]]: 评测对象

### 对比

- [[Ctrl-World]]: 直接基线，是当前 policy-compatible 世界模型 SOTA
- [[Video World Model]]: 同类工作族
- [[World Model]]: 上位概念
- [[Closed-Loop Simulation]]: 与传统物理仿真器 (e.g. [[Isaac Lab]]) 的对比角度

### 方法相关

- [[Action-to-Control Projection]]: 本文创新点 1
- [[Latent History Memory]]: 本文创新点 2
- [[Hallucination-Free Ratio]]: 本文新提出的评测指标
- [[Policy-in-the-Loop]]: 核心范式
- [[LoRA]]: 微调技术
- [[VAE]]: latent 空间编码
- [[LPIPS]]: 视觉相似性指标
- [[Inverse Kinematics]]: 动作投影前置

### 硬件/数据相关

- $\pi_0$ / Pi-0 VLA：被评测策略
- 双臂硬件平台（具体型号论文未披露）
- H20 GPU × 64（pretrain） / × 8（fine-tune）

---

## 速查卡片

> [!summary] PiL-World
> - **核心**: 把 World Model 接口对齐到 VLA 真机闭环（chunk-wise + 多视角 + 反馈循环），让 imagined SR 真正可信
> - **方法**: chunk-wise latent video diffusion + action-derived visual control + latent history memory + success/failure 混合微调
> - **结果**: 平均 $|\Delta SR|$ 63.2% → **12.0%**，HFR 41.5% → **70.1%**，Pearson $\rho{=}0.94$，单步 LPIPS 在 head-view 上降幅高达 −42%
> - **代码**: https://pil-world.github.io

---

*笔记创建时间: 2026-06-08*
