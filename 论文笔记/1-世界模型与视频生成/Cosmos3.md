---
title: "Cosmos 3: Omnimodal World Models for Physical AI"
method_name: "Cosmos3"
authors: [NVIDIA Cosmos Team, Aditi, Niket Agarwal, Arslan Ali, Jon Allen, et al.]
year: 2026
venue: arXiv
tags: [world-model, omnimodal, mixture-of-transformers, video-diffusion, vla, physical-ai, cosmos]
zotero_collection: 1-世界模型与视频生成
image_source: online
arxiv_html: https://arxiv.org/abs/2606.02800
created: 2026-06-05
---

# 论文笔记：Cosmos 3 — Omnimodal World Models for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA（291+ 作者，跨 Cosmos Lab / Robotics / Drive 多团队） |
| 日期 | 2026-06-01（arXiv v1） |
| 项目主页 | https://research.nvidia.com/labs/cosmos-lab/cosmos3 |
| 代码 | https://github.com/nvidia/cosmos |
| 模型权重 | https://huggingface.co/collections/nvidia/cosmos3 |
| 协议 | OpenMDW-1.1（Linux Foundation） |
| 对比基线 | [[Cosmos]] 1/2、[[GAIA-1]]、[[Vista]]、闭源 [[MMDiT]] 系（Imagen/Veo/Sora 类） |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02800) / [PDF](https://arxiv.org/pdf/2606.02800v1) |

---

## 一句话总结

> 一个 [[Mixture-of-Transformers|MoT]] 统一架构,用 language/image/video/audio/action 五模态共享 backbone,把 [[VLM]]、视频生成器、[[World Model|世界模拟器]] 和 [[World-Action Model|WAM]] 收编为一族模型,登顶 Artificial Analysis Text-to-Image / Image-to-Video 与 RoboArena 策略榜。

---

## 核心贡献

1. **统一 omnimodal 架构**: 单个 [[Mixture-of-Transformers|MoT]] backbone 同时处理语言/图像/视频/音频/动作五大模态，输入输出配置高度灵活，不再为每个任务训独立模型。
2. **subsume 四类模型**: 单一框架等价 [[VLM|VLM]] + [[Video Diffusion Model|视频生成器]] + [[World Model|世界模拟器]] + [[World-Action Model|WAM]]，物理 AI 所需的"理解-生成-动作"链路在一个模型里贯通。
3. **SOTA × 4**: 同时拿下 Artificial Analysis 开源 [[Text-to-Image|T2I]] 第一、Image-to-Video 第一，RoboArena 开源策略第一，证明 omnimodal 不是"全都会但都不精"。
4. **完整开源生态**: 代码、checkpoint、合成数据集、评测 benchmark 全部以 OpenMDW-1.1 协议开放，包含 Nano（16B）/Super（64-65B）双尺度，覆盖通用、T2I 专用、I2V 专用、DROID Policy 专用四种 post-trained 变体。
5. **Physical AI 范式**: 把 NVIDIA 的物理 AI 生态（[[Cosmos]]、[[Alpamayo]]、[[OmniDreams]]、AlpaGym）的"基础模型"层正式升级为可同时输出动作和视觉的统一 backbone。

---

## 问题背景

### 要解决的问题

[[Embodied AI|具身智能]] / [[Physical AI|物理 AI]] 系统需要四种能力：
- **理解**: 看懂场景、回答问题（[[VLM]] 范式）
- **生成**: 用语言/图像/动作条件合成视频（[[Video Diffusion Model|视频扩散]] 范式）
- **模拟**: 给定动作预测未来观测（[[World Model|世界模型]] 范式）
- **决策**: 根据观测+指令输出动作（[[VLA|VLA]] / [[World-Action Model|WAM]] 范式）

主流做法是为每种能力训一个独立模型,部署时拼接多模型流水线，存在 **能力割裂**、**部署复杂**、**联合训练困难** 三大问题。

### 现有方法的局限

- **VLM**（[[Qwen3-VL]] 等）擅长理解但不生成像素，无法做闭环 rollout。
- **视频扩散模型**（[[Cosmos]] 1/2、[[Video Diffusion Model|VDM]] 系）擅长高保真生成但不接收动作 / 不输出动作。
- **World-Action Model**（[[OmniDreams]]、[[Alpamayo]]）专精动作-视觉互预测，理解能力弱。
- **跨模态联合训练**: 现有 [[MMDiT]] / Janus 类工作覆盖文本+图像，但还没真正把动作 / 音频也塞进同一架构。

### 本文的动机

NVIDIA 押注一个假设：**用 [[Mixture-of-Transformers|MoT]] 路由,让不同模态共享主干、分配专属 FFN/Attention 参数,就能在一个模型里同时学到所有上述能力,且互相增益**。这与 [[MoE|MoE]] 在 LLM 的成功相呼应，但路由维度从 token 改成了模态。

---

## 方法详解

### 模型架构

Cosmos 3 采用 **omnimodal [[Mixture-of-Transformers|MoT]]** 架构：

- **输入**: 任意模态组合 $\{l, i, v, a, \mathbf{u}\}$（语言 token / 图像 patch / 视频帧 patch / 音频帧 / 动作向量），统一 tokenize 后送进共享 backbone。
- **Backbone**: 共享 [[Self-Attention|self-attention]] + **模态专属 FFN 专家**。每个 transformer block 中：
  - 多模态 token 一起做 attention（保证跨模态依赖建模）
  - FFN 按模态路由到对应专家（保证模态特定特征学习）
- **生成头**:
  - 文本/动作: [[Autoregressive Diffusion|AR head]]，按 next-token 预测
  - 图像/视频/音频: [[Diffusion Model|diffusion head]]，按 velocity / flow 预测
- **输出**: 任意模态组合，例如 $(l, i) \to v$（图文→视频）、$(o_{1:t}, l) \to \mathbf{u}_{t+1}$（VLA 策略）、$(l) \to (i, \text{audio})$（音画联合生成）。

### 模型家族（来自 HuggingFace Collection）

| 变体 | 参数量 | 定位 |
|------|--------|------|
| **Cosmos3-Nano** | 16B | 通用 omnimodal、Physical AI 入门尺度 |
| **Cosmos3-Super** | 64-65B | 前沿尺度，omnimodal SOTA |
| **Cosmos3-Super-Text2Image** | 65B | T2I 专用 post-train（Artificial Analysis 开源 #1） |
| **Cosmos3-Super-Image2Video** | 65B | I2V 专用 post-train（Artificial Analysis 开源 #1） |
| **Cosmos3-Nano-Policy-DROID** | 16B | DROID 操作策略（RoboArena 开源 #1） |

### 核心模块

#### 模块 1: 统一 Tokenizer

**设计动机**: 把异构模态投射到共享 token 空间，使 MoT 能"看"所有模态。

**具体实现**:
- 语言: 标准 BPE token（沿用 LLM 词表）
- 图像/视频: 沿用 Cosmos 1/2 的 [[Cosmos Tokenizer|Cosmos Tokenizer]]，时空压缩到 latent patch
- 音频: 帧级离散 token（音频 codec 类）
- 动作: [[Trajectory Tokenization|动作 tokenization]]，将连续控制量离散化为 codebook 索引

#### 模块 2: Mixture-of-Transformers 路由

**设计动机**: 不同模态的特征分布差异极大,共享 FFN 容易冲突，但完全分开又丢失跨模态对齐。

**具体实现**:
- Attention 层共享（跨模态信息流）
- FFN 层按模态分专家（避免参数干扰）
- LayerNorm / RMSNorm 可选按模态分组
- 与 [[MoE|经典 MoE]] 区别: 路由信号来自模态标签而非 token gating，确定性强、训练稳定

#### 模块 3: Autoregressive + Diffusion 双生成头

**设计动机**: 文本/动作是离散且时序强相关 → AR；图像/视频/音频是连续高维 → diffusion。

**具体实现**:
- 共享 backbone 输出 hidden state 后分两条头:
  - AR head: 下一 token logit
  - Diffusion head: [[Score Function|score]] / velocity 预测，用于连续模态去噪
- 训练时两类 loss 联合优化（见公式 2）

#### 模块 4: 灵活 I/O 配置

任意模态子集做条件、任意模态子集做目标，通过 mask 控制：

- $(l) \to i$: 文生图
- $(l, i) \to v$: 图文生视频
- $(o_{1:t}) \to o_{t+1}$: 视频续写（世界模拟）
- $(o_{1:t}, l) \to \mathbf{u}_{t+1}$: VLA 策略
- $(l) \to (i, \text{audio})$: 多模态生成

---

## 关键公式

### 公式 1: [[Mixture-of-Transformers|MoT 路由]]

$$
h^{(\ell)}_{m,i} = \text{Attn}^{(\ell)}\!\left(\{h^{(\ell-1)}_{m',j}\}_{\forall m',j}\right) + \text{FFN}^{(\ell)}_{m}\!\left(h^{(\ell-1)}_{m,i}\right)
$$

**含义**: 第 $\ell$ 层中，模态 $m$ 的第 $i$ 个 token 的隐藏状态 = 跨所有模态的共享 attention 输出 + 模态专属 FFN 输出。Attention 全局共享让跨模态信息流通,FFN 按模态分专家避免参数干扰。

**符号说明**:
- $m \in \{l, i, v, a, \mathbf{u}\}$: 模态标签（语言/图像/视频/音频/动作）
- $\text{Attn}^{(\ell)}$: 第 $\ell$ 层共享 [[Self-Attention|self-attention]]
- $\text{FFN}^{(\ell)}_m$: 模态 $m$ 在第 $\ell$ 层的专属 FFN 专家
- $h^{(\ell-1)}_{m',j}$: 上一层所有模态所有 token 的隐藏状态

### 公式 2: 联合训练损失

$$
\mathcal{L}_{\text{Cosmos3}} = \lambda_{\text{AR}} \underbrace{\mathbb{E}\!\left[-\log p_\theta(x_t \mid x_{<t}, c)\right]}_{\mathcal{L}_{\text{AR}}} + \lambda_{\text{diff}} \underbrace{\mathbb{E}_{t, x_0, \epsilon, c}\!\left[\| v_\theta(x_t, t, c) - v^* \|_2^2\right]}_{\mathcal{L}_{\text{diff}}}
$$

**含义**: 总损失是离散模态（文本/动作）的自回归 NLL + 连续模态（图像/视频/音频）的 [[Flow Matching|flow-matching / velocity prediction]] 损失的加权和。两路同时反传更新共享 backbone，让 MoT 同时学到"理解"和"生成"。

**符号说明**:
- $x_{<t}$: 自回归历史 token 序列
- $c$: 跨模态条件（语言指令、动作、过去观测等）
- $v_\theta(x_t, t, c)$: 在噪声时刻 $t$ 预测的 velocity 场
- $v^* = \epsilon - x_0$: ground-truth velocity（rectified flow 写法）
- $\lambda_{\text{AR}}, \lambda_{\text{diff}}$: 平衡两路损失的权重

### 公式 3: omnimodal 条件生成（采样接口）

$$
p_\theta(y_{\mathcal{T}} \mid x_{\mathcal{C}}) = \prod_{m \in \mathcal{T}} p_\theta(y_m \mid x_{\mathcal{C}}, y_{\mathcal{T} \setminus m}; \mathcal{M})
$$

**含义**: 给定条件模态集合 $\mathcal{C}$，并行/串行生成目标模态集合 $\mathcal{T}$。同一参数 $\theta$ 通过 mask $\mathcal{M}$ 切换任务（T2I / I2V / VLA / world rollout），实现"一个模型多种 I/O"。

**符号说明**:
- $\mathcal{C} \subseteq \{l, i, v, a, \mathbf{u}\}$: 条件模态集
- $\mathcal{T} \subseteq \{l, i, v, a, \mathbf{u}\}$: 目标模态集，与 $\mathcal{C}$ 互斥
- $\mathcal{M}$: I/O 模式 mask，决定哪些位置参与生成、哪些作为条件

### 公式 4: VLA 推理（策略模式）

$$
\mathbf{u}_{t+1:t+H} = \arg\max_{\mathbf{u}} \; p_\theta\!\left(\mathbf{u} \mid o_{t-K+1:t}, l\right)
$$

**含义**: 在 Cosmos3-Nano-Policy-DROID 配置下，模型把过去 $K$ 帧观测 $o_{t-K+1:t}$ 和语言指令 $l$ 作为条件，输出长度 $H$ 的动作 chunk，实现 [[Action Chunking|动作分块]] 的 [[VLA|视觉-语言-动作]] 策略。

**符号说明**:
- $o_{t-K+1:t}$: 过去 $K$ 帧视觉观测
- $l$: 自然语言任务指令
- $H$: 动作 chunk 长度
- $\mathbf{u}_{t+1:t+H}$: 未来 $H$ 步动作序列

---

## 关键图表

### Figure 1: Omnimodal MoT 架构总览

![Figure 1](https://raw.githubusercontent.com/NVIDIA/cosmos/main/cookbooks/cosmos3/cosmos3-model-architecture.png)

**说明**: Cosmos 3 的核心 [[Mixture-of-Transformers|MoT]] 架构。五个模态（语言/图像/视频/音频/动作）的 token 共享 transformer backbone 的 attention，但 FFN 按模态路由到专属专家。生成头分两路：离散模态用 [[Autoregressive Diffusion|AR head]]，连续模态用 [[Diffusion Model|diffusion head]]。一个模型支持任意 I/O 配置，覆盖 T2I / I2V / 世界模拟 / VLA 策略等任务。

### Figure 2: Physical AI 应用场景（项目主页 teaser）

![Figure 2](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/icons/t.png)

**说明**: Cosmos 3 覆盖的 Physical AI 应用谱系——固定翼测绘无人机、激光切割龙门、汽车装配机器人、双臂引擎吊装、外科手术臂、冷库叉车、AGV 货车、机器人系鞋带等。统一 backbone 让同一组权重支撑从无人机到外科机器人的多形态部署。

> 注：图 2 链接为项目主页 teaser 集合的代表图标；如需完整动图可访问 https://research.nvidia.com/labs/cosmos-lab/cosmos3 查看。

### Table 1: 模型家族概览

| 变体 | 参数量 | 任务定位 | 第三方排名 |
|------|--------|----------|------------|
| Cosmos3-Nano | 16B | 通用 omnimodal | — |
| Cosmos3-Super | 65B | 前沿 omnimodal | — |
| **Cosmos3-Super-T2I** | 65B | Text-to-Image | **Artificial Analysis 开源 #1** |
| **Cosmos3-Super-I2V** | 65B | Image-to-Video | **Artificial Analysis 开源 #1** |
| **Cosmos3-Nano-Policy-DROID** | 16B | DROID Robot Policy | **RoboArena 开源 #1** |

**说明**: 单一 omnimodal pretrain 后 post-train 出 5 个变体，三个专用变体均拿到对应赛道开源 SOTA，证明 omnimodal 不会牺牲单任务上限。

### Table 2: 任务能力矩阵（与单一专精模型对比）

| 任务 | [[VLM|VLM]] (e.g. Qwen3-VL) | [[Video Diffusion Model|VDM]] (e.g. Cosmos 2) | [[World-Action Model|WAM]] (e.g. OmniDreams) | [[VLA|VLA]] (e.g. Alpamayo) | **Cosmos 3** |
|------|---|---|---|---|---|
| 视觉理解 | + | − | − | partial | **+** |
| 文生图 | − | partial | − | − | **+** |
| 图生视频 | − | + | partial | − | **+** |
| 动作条件视频 | − | − | + | − | **+** |
| 语言-动作策略 | − | − | partial | + | **+** |
| 音频生成 | − | − | − | − | **+** |

**说明**: Cosmos 3 把过去需要 4-5 个专精模型拼接的物理 AI 链路收编进单一 backbone。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Cosmos 通用视频先验 | ~20M 小时（继承 Cosmos 1/2） | 通用世界视觉 | pretrain |
| PhysicalAI-WorldModel-Synthetic-* | 5 个合成场景包（仓储/驾驶/具身/人形/物理交互） | NVIDIA Omniverse 合成 | mid-train |
| LIBERO_LeRobot_v3 | LIBERO 任务集，LeRobot v3 格式 | 机器人 manipulation | post-train（policy） |
| BridgeData2_LeRobot_v3 | BridgeV2 重打包 | 机器人 manipulation | post-train（policy） |
| DROID | ~76K 操作 episode | 大规模真机数据 | post-train（policy） |

### 实现细节（公开信息）

- **模型规模**: Nano 16B / Super 65B（参考 HuggingFace 模型卡）
- **协议**: OpenMDW-1.1（Linux Foundation 新出的开放权重协议）
- **生态**: 与 Cosmos Framework、AlpaGym、NeMo 训练栈深度集成
- **硬件**: 推断需 H100/B200 级 GPU，具体训练规模在 `inference_benchmarks.md` 中

### 关键实验结果

1. **Artificial Analysis Text-to-Image**: Cosmos3-Super-Text2Image 排名 **开源模型第一**（截至论文写作时）
2. **Artificial Analysis Image-to-Video**: Cosmos3-Super-Image2Video 排名 **开源模型第一**
3. **RoboArena**: Cosmos3-Nano-Policy-DROID 排名 **最佳开源策略模型**
4. **能力统一**: 同一 backbone 同时通过 VLM 评测（视觉问答）、视频生成评测（FVD/视觉质量）、机器人策略评测（DROID 成功率），全部进入开源 SOTA 行列

---

## 批判性思考

### 优点

1. **统一性极强**: 真正做到 language + image + video + audio + action 五模态共享 backbone，比 Janus / Chameleon 类只覆盖 2-3 模态的工作更彻底。
2. **不牺牲单任务**: 三个 post-trained 变体都拿到对应赛道开源 SOTA，反驳了"通用模型必平庸"的常见担忧。
3. **完整开源 + 工业级生态**: 代码、模型、合成数据集、benchmark 都开放，且与 NVIDIA Omniverse / Isaac / Cosmos Framework 闭环，对工业用户极友好。
4. **MoT 路由设计简洁**: 用模态确定性路由代替 [[MoE|MoE]] 的 learned gating，训练更稳定，避开"专家坍缩"等常见 MoE 病。

### 局限性

1. **作者列表 291+ 暗示工程巨制**: 复现门槛极高,小团队几乎不可能从零训。
2. **参数量大**: Super 65B 推理对显存要求高，端侧/嵌入式部署需进一步蒸馏 / 量化。
3. **MoT 是否真的优于单模态专家拼接,论文需更系统的消融**: 单看 SOTA 排名,无法直接归因于 MoT 本身还是数据/工程优势。
4. **动作模态评测局限于 DROID/LeRobot 任务**: 真实部署到双足、双臂、灵巧手平台时的迁移性仍待验证。
5. **音频模态在 Physical AI 上下文的必要性**未充分论证（机器人交互真的需要联合音视频生成吗?）。

### 潜在改进方向

1. **小尺度 Nano-Mini**（<3B）做边缘部署。
2. **加入触觉 / IMU / proprioception** 作为第六、第七模态。
3. **Online RL 微调**: 与 AlpaGym 闭环搭配，做 [[World-Action Model|WAM]] 的 RLHF / RLAIF。
4. **跨形态泛化**: 同一 policy head 服务多种 embodiment（人形 + 移动 + 操作）。

### 可复现性评估

- [x] 代码开源（GitHub: nvidia/cosmos）
- [x] 预训练模型（HuggingFace: nvidia/cosmos3）
- [x] 训练框架（cosmos-framework + NeMo）
- [x] 合成数据集（HuggingFace 5 个 PhysicalAI 数据集）
- [ ] 完整训练复现（算力门槛在数千 H100·月）

---

## 关联笔记

### 基于

- [[Cosmos]]: 直接前身,沿用其视频先验与 tokenizer
- [[Mixture-of-Transformers|MoT 架构]]: 借鉴 LLM MoE 思想，路由维度改为模态

### 应用 / 衍生

- [[OmniDreams]]: 同一周期 NVIDIA 工作,在 Cosmos 视频骨干上做驾驶 WAM；Cosmos 3 提供了"omnimodal backbone"层
- [[Alpamayo]]: NVIDIA Drive VLA,可视作 Cosmos 3 在驾驶域的 specialization

### 方法相关

- [[Mixture-of-Transformers]]: 核心架构
- [[Autoregressive Diffusion]]: 离散+连续联合生成范式
- [[World-Action Model]]: VLA / 策略输出能力的理论框架
- [[Video Diffusion Model]]: 视频生成头的基础

### 对比

- [[GAIA-1]]、[[Vista]]、[[DriveDreamer]]: 单任务驾驶世界模型，被 Cosmos 3 一统
- [[VLA]] 系（[[Alpamayo]]）: 单任务策略模型，被 Cosmos 3 在 RoboArena 超越

### 硬件 / 数据

- DROID / LIBERO / BridgeV2: post-train 用机器人数据
- PhysicalAI-WorldModel-Synthetic-* 系列: NVIDIA 自家合成数据

---

## 速查卡片

> [!summary] Cosmos 3: Omnimodal World Models for Physical AI
> - **核心**: 一个 [[Mixture-of-Transformers|MoT]] 主干同时吃/吐 language/image/video/audio/action 五模态
> - **方法**: 共享 attention + 模态专属 FFN 专家 + AR&Diffusion 双生成头 + 灵活 I/O mask
> - **结果**: 开源 T2I / I2V / RoboArena 策略三项第一,Nano 16B + Super 65B 双尺度
> - **代码**: https://github.com/nvidia/cosmos
> - **意义**: 把 VLM + 视频生成 + 世界模拟 + VLA 收编为单一 backbone,成为 NVIDIA 物理 AI 生态新底座

---

*笔记创建时间: 2026-06-05*
