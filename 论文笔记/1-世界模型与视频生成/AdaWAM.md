---
title: "Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning"
method_name: "AdaWAM"
authors: [Yinzhou Tang, Jingbo Xu, Yu Shang, Zihao Song, Chen Gao, Wei Wu, Yong Li]
year: 2026
venue: arXiv
tags: [world-action-model, adaptive-reasoning, vla, video-diffusion, dynamic-routing, libero, robotwin]
zotero_collection: 1-世界模型与视频生成
image_source: local
arxiv_html: https://arxiv.org/abs/2606.07089
created: 2026-06-09
---

# 论文笔记：AdaWAM (Dreaming when Necessary)

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学（Tsinghua University, BNRist, DCST） |
| 日期 | 2026-06-05 |
| 项目主页 | https://adawam.github.io/ |
| 对比基线 | [[Pi05|π₀.₅]]、[[VLA|X-VLA]]、Motus、Fast-WAM、LingBot-VA、[[Pi0|π₀]]、OpenVLA、CoT-VLA、ACoT-VLA、MM-ACT、GO-1、GigaBrain-0 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07089) / [PDF](https://arxiv.org/pdf/2606.07089) / [Project](https://adawam.github.io/) |

---

## 一句话总结

> 给 [[World-Action Model|WAM]] 装一个轻量动态路由器，让 [[Video Diffusion Model|视频扩散]] 只在长程任务切换需要"文本推理"或精细操作需要"视觉做梦"时才被触发，其余时刻直接走 action-only，把推理算力花在刀刃上。

---

## 核心贡献

1. **"按需做梦"范式**: 提出 [[Adaptive Multi-Modal Reasoning|自适应多模态推理]] —— 用一个动态路由器 $\mathcal{R}_\psi$ 在每个 action chunk 上独立决定是否触发文本推理 `<TR>` / 视觉推理 `<VR>`，避免现有 WAM "每步都做全量视频预测"的算力浪费。
2. **两条推理路径分工**: 文本推理负责**任务转换时的高层语义**（下一 subtask 是什么）；视觉推理负责**精细操作时的低层物理**（未来几帧 latent 长什么样），二者职责正交。
3. **轨迹启发式标注 + VLM 语义校验**: 设计了一条不依赖人类标注的 [[Multimodal Reasoning Data Annotation|多模态推理数据标注 pipeline]]，由 end-effector 运动模式定位候选窗口，再用 Qwen3-VL-8B 做语义验证，自动产出 `<TR>` / `<VR>` 监督标签。
4. **两阶段训练**: Stage 1 联合训练 Video-Action DiT 和路由器（50K 步），Stage 2 冻结主干微调文本推理模块（10K 步），避免灾难性遗忘。
5. **SOTA + 推理高效**: [[LIBERO]]-Long **99.1%**（超过 X-VLA 1.5pt，超过 Fast-WAM 3.9pt），[[RoboTwin]] 2.0 Hard 子集 **88.43%**，而每步推理延迟与 action-only Fast-WAM 持平；真机 Clean Up Trash 70% / Wipe Table 60% 均最高。

---

## 问题背景

### 要解决的问题

[[World-Action Model|WAM]] 范式（如 Fast-WAM、Motus、π₀.₅）通过预测未来视觉帧作为"先想象再行动"的中间步骤，在长程操作上明显优于纯 action-only [[VLA]]。但**所有现有 WAM 都是"每步都做梦"**：不管这一步是简单平移还是关键抓取转换点，都要跑一遍完整的 [[Video Diffusion Model|视频扩散]]，单步延迟显著高于 action-only。

### 现有方法的局限

- **Video-action joint prediction**（Fast-WAM、Motus）：每个 action chunk 都生成未来 K 帧 latent，延迟高且大部分帧的"想象"对动作输出贡献甚微。
- **Action-only prediction**（π₀.₅、X-VLA）：放弃了视觉预测，长程任务（[[LIBERO]]-Long、[[RoboTwin]]-Hard）成功率显著下降。
- 两者都用了**单一模态推理路径**，没有考虑"任务转换需要语义、精细操作需要视觉"这种分工——而这恰恰是人类驾驶/操作的真实模式。

### 本文的动机

观察：在一段 manipulation 轨迹里，**真正需要"做梦"的步骤可能只有 10–30%**——通常是子任务切换点（"放下抹布去拿杯子"）或精细对准点（"把杯耳挂到挂钩上"）。如果能学一个轻量路由器在每个 chunk 上判断"现在该不该做梦、做哪种梦"，就能用 1 倍延迟拿到接近 N 倍的能力。

---

## 方法详解

### 模型架构

AdaWAM 采用**四模块解耦 + 动态路由**架构：

- **输入**: 全局指令 $l$ + 历史视觉 latent $z_{\le t}$ + 当前 subtask 描述 $c_t$
- **Backbone**:
  - 视觉/动作: 基于 Wan2.2-5B 蒸馏的 Video-Action [[DiT|Diffusion Transformer]]（动作侧压到 1B、hidden=1024）
  - 文本: [[VLM|Qwen3-VL-4B]]
- **核心模块**:
  - [[Video-Action DiT]] $\mathcal{M}_\theta + \pi_\phi$ 共享 backbone，分别预测未来视觉 latent 和动作 chunk
  - [[Text Reasoning Module]] $\mathcal{V}_\omega$ 自回归预测下一 subtask
  - [[Dynamic Router]] $\mathcal{R}_\psi$ 输出二进制 `<TR>`, `<VR>` 决策 token
- **输出**: [[Action Chunking|chunked actions]] $a_{t:t+H}$（$H$ 为 chunk 长度）
- **总参数**: 视觉/动作约 5B + 文本 4B ≈ 9B（推理时按路由稀疏激活）

### 核心模块

#### 模块 1: Multimodal Reasoning Data Annotation Pipeline

**设计动机**: 路由器需要监督标签——"这一步该不该做梦"——但人类不可能逐 chunk 标注。

**具体实现**:
- **Trajectory-guided Subtask Annotation**: 提取 end-effector 位移、姿态变化、夹爪开闭、局部运动方差等运动学特征，自动定位"可能的 subtask 转换窗口"；再用 Qwen3-VL-8B 在视频上做语义验证，确认 subtask 实际完成；强制 subtask 序列单调递增，避免标注回环。
- **Motion-Based Fine Manipulation Labeling**: 把机器人状态序列映射成运动模式串，区分 *grasping / releasing / in-hand adjustment / local pose refinement* 四类，将后两类标为 `<VR>=1`，前两类标为关键点但不强制 `<VR>`。

#### 模块 2: Video-Action DiT

**设计动机**: 复用同一 [[Flow Matching]] 主干同时建模"世界"和"动作"，参数效率高（[[World-Action Model|WAM]] 的标准设计）。

**具体实现**:
- **VideoDiT** $\mathcal{M}_\theta$: 输入历史 latent $z_{\le t}$ 和当前 subtask $\tilde{c}_t$，预测未来 K 帧 latent $z_{t+1:t+K}$。
- **ActionDiT** $\pi_\phi$: 输入打包多模态集 $\mathcal{S}_t = \{z_{\le t}, \tilde{c}_t, \tilde{z}_f\}$，输出动作 chunk $a_{t:t+H}$。
- 两侧均采用 [[Flow Matching]] 训练目标。

#### 模块 3: Text Reasoning Module

**设计动机**: 任务转换是**语义级**的（"我已经把杯子放好，下一步该擦桌子"），用 [[VLM]] 更自然。

**具体实现**:
- 基于 Qwen3-VL-4B，输入 $(z_{\le t}, l)$ 自回归预测下一 subtask 描述 $c_{t+1}$。
- 只在 `<TR>=1` 时被调用，**否则沿用上一时刻的 $c_t$**。
- Stage 2 训练，主干和路由器冻结。

#### 模块 4: Dynamic Router

**设计动机**: 路由器必须**轻**（不能反噬延迟），但又要看懂视觉+语言上下文。

**具体实现**:
- 输入上下文向量 $C_t = [v_t \,\|\, e_l \,\|\, e_{c_t}]$，其中 $v_t$ 为 $z_t$ 的池化嵌入、$e_l / e_{c_t}$ 分别为全局指令和当前 subtask 的文本嵌入。
- 输出两个离散 token: `<TR>` 和 `<VR>`，组合出 4 种推理模式：
  - `<TR>=0, <VR>=0`: 仅 action（最快，平移阶段）
  - `<TR>=0, <VR>=1`: action + 视觉想象（精细对准）
  - `<TR>=1, <VR>=0`: action + 文本规划（任务切换）
  - `<TR>=1, <VR>=1`: 全量推理（关键转折）
- 由启发式标签 $\hat{y}_k$（见模块 1）通过 [[Binary Cross-Entropy|BCE]] 监督。

#### 模块 5: Test-Time Adaptive Inference

**设计动机**: 推理时每个 action chunk 独立路由，避免一刀切。

**具体实现**:
1. 路由器先看 $C_t$ 给出 `<TR>`, `<VR>`；
2. 若 `<TR>=1`，调 $\mathcal{V}_\omega$ 更新 $\tilde{c}_t$；否则继承 $c_t$；
3. 若 `<VR>=1`，调 $\mathcal{M}_\theta$ 生成未来 latent $\tilde{z}_f$；否则置空；
4. ActionDiT 在组装好的 $\mathcal{S}_t$ 上采样 $a_{t:t+H}$。

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}\Big[\,\|v_\theta(z_\tau, \cdot) - (z_1 - z_0)\|^2 \;+\; \|v_\phi(a_\tau, \cdot) - (a_1 - a_0)\|^2\,\Big]
$$

**含义**: 视觉 latent 和动作 chunk 同时用 [[Flow Matching]] 训练 —— 网络 $v_\theta / v_\phi$ 学习把"噪声端 $z_0/a_0$"沿直线流到"数据端 $z_1/a_1$"的速度场。这是 Video-Action DiT 共享的核心 loss。

**符号说明**:

- $\tau \sim \mathcal{U}(0, 1)$: 流匹配的连续时间
- $z_\tau = \tau z_1 + (1-\tau) z_0$: 视觉 latent 的线性插值（$z_0 \sim \mathcal{N}(0, I)$, $z_1$ 为真实未来 latent）
- $a_\tau = \tau a_1 + (1-\tau) a_0$: 动作的线性插值
- $v_\theta, v_\phi$: VideoDiT / ActionDiT 速度场预测网络

### 公式 2: [[Joint Routing Loss|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} \;+\; \lambda \sum_{k \in \{\text{text},\, \text{video}\}} \mathcal{L}_{\text{BCE}}\big(y_k,\, \hat{y}_k\big)
$$

**含义**: Stage 1 训练把流匹配 loss 与路由器 [[Binary Cross-Entropy|BCE]] loss 加权相加，让生成主干和路由决策共同优化。

**符号说明**:

- $\mathcal{L}_{\text{BCE}}$: 二分类交叉熵
- $y_k \in \{0, 1\}$: 路由器对模态 $k$ 的预测概率
- $\hat{y}_k$: 启发式标注得到的金标准路由标签（来自 trajectory cue + VLM 语义验证）
- $\lambda$: 路由 loss 权重（论文未给出具体数值，量级与 $\mathcal{L}_{\text{FM}}$ 平衡）

### 公式 3: [[Text Reasoning Module|文本推理路由更新]]

$$
\tilde{c}_t = \begin{cases}
\mathcal{V}_\omega(z_{\le t},\, l) & \text{if } \langle \text{TR}\rangle = 1 \\
c_t & \text{if } \langle \text{TR}\rangle = 0
\end{cases}
$$

**含义**: `<TR>` 为 1 时调用 [[VLM|Qwen3-VL-4B]] 自回归推理出新的 subtask 描述，否则沿用上一 step。**这是计算最贵的开关**——大部分 chunk 都走下分支。

**符号说明**:

- $\tilde{c}_t$: 路由后实际使用的 subtask token 序列
- $\mathcal{V}_\omega$: 文本推理模块（Qwen3-VL-4B）
- $c_t$: 上一时刻的 subtask（继承自上次 `<TR>=1` 的输出）

### 公式 4: [[Video Diffusion Model|视觉想象路由开关]]

$$
\tilde{z}_f = \begin{cases}
\mathcal{M}_\theta(z_{\le t},\, \tilde{c}_t) & \text{if } \langle \text{VR}\rangle = 1 \\
\varnothing & \text{if } \langle \text{VR}\rangle = 0
\end{cases}
$$

**含义**: `<VR>=1` 时调用 VideoDiT 生成未来 latent $\tilde{z}_f$ 作为 ActionDiT 的额外条件；否则把视觉想象置空，由动作主干直接出 chunk。**这是把 video diffusion 推理算力按需打开的核心**。

**符号说明**:

- $\tilde{z}_f$: 未来视觉 latent（若开启），输入给 ActionDiT
- $\mathcal{M}_\theta$: VideoDiT（[[Wan2.1|Wan2.2]] 5B 蒸馏版）
- $\varnothing$: 空集，表示不注入视觉条件

### 公式 5: [[Action Chunking|动作 chunk 采样]]

$$
a_{t:t+H} \,\sim\, \pi_\phi(\,\cdot \,\mid \,\mathcal{S}_t\,), \quad \mathcal{S}_t = \{z_{\le t},\, \tilde{c}_t,\, \tilde{z}_f\}
$$

**含义**: ActionDiT 在最终组装的多模态集 $\mathcal{S}_t$ 上采样动作 chunk。$\mathcal{S}_t$ 的内容会根据路由决策动态变化——这是 AdaWAM 推理稀疏性的来源。

**符号说明**:

- $a_{t:t+H}$: 从 $t$ 起 $H$ 步的动作 chunk
- $\mathcal{S}_t$: 自适应多模态条件集（视觉历史 + 路由后 subtask + 可选视觉想象）
- $\pi_\phi$: ActionDiT 策略网络（1B 压缩版）

---

## 关键图表

### Figure 1: Paradigm Comparison / 三种 WAM 范式对比

![[AdaWAM_fig1_paradigm.png]]

**说明**: 对比 (a) 传统 video-action joint prediction WAM（每步都做完整视觉预测）、(b) action-only WAM（完全省去视觉预测）、(c) **AdaWAM 自适应多模态推理**（路由器按需触发文本/视觉推理）。可以看到 AdaWAM 在大部分平移步骤跳过视觉预测，在关键转折点才开启 `<TR>` 或 `<VR>`，实现 *"必要时才做梦"*。

### Figure 2: Annotation Pipeline / 多模态推理数据标注流程

![[AdaWAM_fig2_annotation.png]]

**说明**: 数据标注 pipeline 详图。左侧 *Trajectory-guided Subtask Annotation*: 从机器人轨迹提取 end-effector 运动学特征 → 定位 subtask 候选窗口 → 用 [[VLM|Qwen3-VL-8B]] 做语义校验 → 输出 `<TR>` 标签；右侧 *Motion-Based Fine Manipulation Labeling*: 把机器人状态映射到 4 类运动模式（grasping / releasing / in-hand adjustment / local refinement）→ 后两类标为 `<VR>=1`。整条 pipeline 完全自动，无需人工标 chunk 级路由。

### Figure 3: AdaWAM Model Architecture / 总体架构

![[AdaWAM_fig3_architecture.png]]

**说明**: AdaWAM 完整架构。中央是共享的 Video-Action [[DiT|Diffusion Transformer]] backbone（VideoDiT + ActionDiT 通过 [[Flow Matching]] 联合训练）；左上是 [[Text Reasoning Module|Qwen3-VL-4B 文本推理模块]]；左下是 [[Dynamic Router|动态路由器]]，吃 $C_t = [v_t \|\| e_l \|\| e_{c_t}]$ 输出 `<TR>` / `<VR>`；右侧是 [[Action Chunking|chunk 化的动作输出]]。两个 router gate 把 $\mathcal{V}_\omega$ 和 $\mathcal{M}_\theta$ 的调用稀疏化。

### Figure 4: Real-World Long-Horizon Task / 真机长程任务可视化

![[AdaWAM_fig4_realworld_clean.png]]

![[AdaWAM_fig4_realworld_wipe.png]]

**说明**: 真机部署的两个长程任务 (a) *Clean Up Trash on Table*——AdaWAM 70% vs π₀.₅ 60% vs Fast-WAM 30%；(b) *Wipe Table Clean*——AdaWAM 60% vs π₀.₅ 50% vs X-VLA/Motus 20%。可以看到 AdaWAM 在多步骤抓取-移动-释放-擦拭这种长程组合任务上明显跑赢。

### Figure 5: Fine-Grained Case Study (HangingMug) / 精细操作案例对比

![[AdaWAM_fig5_hangingmug_case.png]]

**说明**: HangingMug（把杯耳挂到挂钩上）的成败对比。上排 *Action-Only WAM* 在最后挂钩对准时由于缺乏视觉想象，没能精确把杯耳套进挂钩；下排 *AdaWAM* 在该关键步骤触发 `<VR>=1`，VideoDiT 预测出未来几帧的视觉对准目标，ActionDiT 在视觉先验下精确完成挂钩。直观说明 *adaptive visual reasoning* 在精细对准上的价值。

### Figure 6a: Inference Time Analysis - StackThreeBowls / 堆三碗任务的精度-延迟曲线

![[AdaWAM_fig6a_stackbowls_latency.png]]

**说明**: 横轴为单步推理延迟，纵轴为成功率。AdaWAM 位于左上角（高成功率 + 与 action-only Fast-WAM 接近的延迟），明显优于"每步都做梦"的 Motus（高延迟）。证明路由器的稀疏调度并未带来明显延迟开销。

### Figure 6b: Inference Time Analysis - PutObjectCabinet / 放物体入柜的精度-延迟曲线

![[AdaWAM_fig6b_putcabinet_latency.png]]

**说明**: 另一组 RoboTwin 任务上的精度-延迟散点。AdaWAM 同样占据帕累托最优位置（96% 成功率 / 与 action-only 接近的延迟）。两图共同说明：*"按需做梦"在不牺牲延迟的前提下显著提升成功率*。

### Table 1: LIBERO 完整对比

| Model | Spatial | Object | Goal | Long | Overall |
|-------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| [[Pi0\|π₀]] | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| ACoT-VLA | 98.6 | 99.0 | 99.4 | 97.0 | 98.5 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 88.0 | 95.0 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | 98.2 | **100.0** | 97.0 | 95.2 | 97.6 |
| AdaWAM w/o V.R. | 97.5 | 99.4 | 96.8 | 96.6 | 97.6 |
| AdaWAM w/o T.R. | — | — | — | 97.4 | — |
| **AdaWAM** | 98.0 | 99.6 | 97.1 | **99.1** | **98.5** |

**说明**: AdaWAM 在 LIBERO-Long 子集上拿到 99.1%——这是真正考验长程推理能力的 suite，比 X-VLA (97.6)、Fast-WAM (95.2)、Motus (97.6) 都明显高。其余子集已接近饱和（97–100%）。

### Table 2: RoboTwin 2.0 Clean Environment 完整对比

| Task | GO-1 | π₀ | π₀.₅ | X-VLA | LingBot-VA | Motus | Fast-WAM | AdaWAM w/o V.R. | AdaWAM w/o T.R. | **AdaWAM** |
|------|------|----|----|-------|-----------|-------|---------|------|------|------|
| HangingMug | 0 | 14 | 18 | 23 | 40 | 38 | 58 | 56 | 56 | **59** |
| PickDiverseBottles | 61 | 69 | 81 | 58 | 89 | 90 | 80 | 79 | 81 | **87** |
| PutObjectCabinet | 60 | 85 | 80 | 46 | 80 | 88 | 94 | 92 | 91 | **96** |
| RotateQRCode | 22 | 74 | 89 | 34 | 96 | 89 | 93 | 95 | 96 | 94 |
| ScanObject | 1 | 55 | 72 | 14 | 96 | 67 | 89 | 91 | 88 | **92** |
| StackBowlsThree | 4 | 77 | 77 | 76 | **100** | 79 | 80 | 77 | 82 | **100** |
| StampSeal | 19 | 46 | 79 | 76 | **96** | 93 | 90 | 93 | 92 | 91 |
| **Hard SR** | 23.86 | 60.00 | 70.86 | 46.71 | 85.29 | 77.71 | 83.43 | 83.29 | 83.71 | **88.43** |
| **Overall SR** | 37.80 | 65.92 | 82.74 | 72.80 | 92.90 | 88.66 | 91.88 | 91.31 | 91.88 | **93.11** |

**说明**: Hard subset 上 AdaWAM 88.43%，显著领先 Fast-WAM 5pt、LingBot-VA 3pt。HangingMug（精细对准）和 PutObjectCabinet（长程）这两类任务受益最大，对应"视觉做梦"和"文本规划"两条路径各自的强项。Random Environment（视觉扰动）下趋势一致，AdaWAM 维持 ≈ 91% Overall SR。

### Table 3: 真机 Real-World 实验

| Task | π₀.₅ | X-VLA | Motus | GigaBrain-0 | Fast-WAM | AdaWAM w/o V.R. | AdaWAM w/o T.R. | **AdaWAM** |
|------|------|-------|-------|-------------|---------|------|------|------|
| Clean Up Trash On Table | 60 | 60 | 50 | 10 | 30 | 30 | 50 | **70** |
| Wipe Table Clean | 50 | 20 | 20 | 10 | 50 | 60 | 60 | **60** |

**说明**: 两个真机长程任务上 AdaWAM 均最高。注意 Fast-WAM 在 Clean Up Trash 上只有 30%（每步都做梦反而拖累实时性），而 AdaWAM 把 video diffusion 用在刀刃上反而效果更好——这是论文最强的 *real-world* 证据。

### Table 4: Compositional Generalization / 未见 subtask 组合泛化

| Model | AdaWAM w/o T.R. | AdaWAM w/o V.R. | **AdaWAM** |
|-------|------|------|------|
| Unseen Task (soup&butter) | 3 | 38 | **61** |

**说明**: 训练只见过 *soup&cheese* + *cheese&butter*，测试 *soup&butter* 组合。**去掉文本推理后成功率塌缩到 3%**——证明文本推理对组合泛化是不可或缺的；视觉推理也提供 38% → 61% 的额外增益。这是消融实验里最戏剧化的发现，强烈支持"两条路径职责正交"的设计假设。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 suites × 各 10 任务 | Spatial / Object / Goal / Long 四向泛化 | 训练 + 测试 |
| [[RoboTwin]] 2.0 | 50 任务（7 个 Hard） | 双手 manipulation, Clean/Random | 训练 + 测试 |
| 真机数据 | Clean Up Trash, Wipe Table | 两个长程任务 | 真机部署评测 |

### 实现细节

- **VideoDiT**: 基于 [[Wan2.1|Wan2.2]]-5B 蒸馏（图像编码 + DiT backbone）
- **ActionDiT**: 1B 压缩版，hidden dim 1024
- **Text Reasoning**: [[VLM|Qwen3-VL-4B]]
- **路由器**: 轻量 MLP head 输出二进制 token `<TR>`, `<VR>`
- **Stage 1（联合训练）**: 50,000 步, lr 3×10⁻⁵, weight decay 0.005, [[Flow Matching|流匹配]] + [[Binary Cross-Entropy|BCE]]
- **Stage 2（文本推理微调）**: 10,000 步, lr 1×10⁻⁵, weight decay 0.01, 冻结 backbone 和 router, 监督 subtask 切换点的 NLL
- **硬件**: 8 × NVIDIA A100 (80GB)
- **总训练步数**: 60K

### 可视化结果

论文提供：

- HangingMug 关键挂钩对准帧的失败/成功对比（Fig 5）
- 真机 Clean Table 与 Wipe Table 完整 rollout 序列（Fig 4）
- 路由器在长程任务里 `<TR>` / `<VR>` 触发的时间分布（论文正文描述：触发率 ≈ 15–30% 区间）

---

## 批判性思考

### 优点

1. **范式 insight 干净**: 把"该不该做梦"这件事变成一个**显式的、可学习的、二分类问题**，比之前隐式让模型自己学要可控得多——这个 framing 应该会被很多后续工作借鉴。
2. **职责正交的双路径**: 文本管 subtask 切换、视觉管精细对准，两条路径各自的消融实验（去 T.R. → 组合泛化崩塌，去 V.R. → 精细操作下降）都明确支持设计假设。
3. **自动化数据标注**: trajectory cue + VLM 校验的 pipeline 不依赖人类逐 chunk 标注，是个真正能 scale 的思路。
4. **延迟-精度帕累托**: Fig 6 的两组散点很有说服力——把 video diffusion 推理算力*按需*打开，避免了 WAM "高精度但慢"的传统 trade-off。
5. **真机验证**: 不只跑 sim，两个真机长程任务都打了出来，且对照基线齐全。

### 局限性

1. **路由训练依赖启发式标签**: $\hat{y}_k$ 是 trajectory cue + VLM 检测的产物，启发式本身可能有偏差（论文 Limitations 也承认）。若启发式 miss 了某些"应该做梦"的微妙时刻，路由器会系统性遗漏。
2. **观测仅限 RGB**: 论文自己也指出，没有引入深度/点云/触觉，对接触丰富且几何复杂的任务（如插孔、布料折叠）会受限。
3. **路由粒度 = action chunk**: 二进制门控只能 chunk-level 切换，做不到"chunk 内部前 3 步要视觉、后 5 步不要"的精细调度。
4. **路由器小，但 VLM 4B 调一次也不便宜**: `<TR>=1` 触发的 Qwen3-VL-4B 推理仍是 4B 参数的 forward，平均延迟受 `<TR>` 触发频率影响显著。论文未展示 `<TR>` 触发率与延迟的细粒度关系。
5. **训练标签缺少 RL fine-tuning**: 完全监督训练，缺少基于环境 reward 的策略性调优（Limitations 第二点已指出）。
6. **跨实体迁移**: 仅在固定 bimanual 平台测试，没展示 humanoid / mobile manipulator 等异构 embodiment 的 transfer。

### 潜在改进方向

1. **路由器加 RL fine-tune**: 用环境成功率作 reward，让路由器自己学"哪些步真的值得做梦"，减少对启发式标签的依赖。
2. **细到 token 的路由**: 把二进制 `<VR>` 推广到 *选择性视觉条件*（哪些 token 注入 $\tilde{z}_f$），进一步压缩 video diffusion 算力。
3. **多模态观测**: 加入触觉或点云，扩展到接触丰富任务。
4. **跨实体 router transfer**: 训练 router 时 randomize embodiment，使其在 humanoid / mobile base 上零样本可用。
5. **路由蒸馏**: 把"AdaWAM 路由序列"蒸馏给一个静态调度策略，进一步去掉路由器自身的推理开销。

### 可复现性评估

- [x] 项目主页存在（adawam.github.io，论文称代码+demo 公开）
- [ ] 预训练模型公开情况待项目主页释放
- [x] 训练细节完整（两阶段、学习率、weight decay、步数齐全）
- [x] 数据集可获取（[[LIBERO]] + [[RoboTwin]] 2.0 都是公开 benchmark）
- [ ] 真机数据未公开
- [x] 主干模型可获取（[[Wan2.1|Wan2.2]] / Qwen3-VL 均公开）

---

## 关联笔记

### 基于

- [[World-Action Model]]: AdaWAM 是 WAM 范式的"自适应"升级版
- [[Video Diffusion Model]]: VideoDiT 的基础架构
- [[Flow Matching]]: 视觉与动作侧共享的训练目标
- [[VLA]]: 文本推理路径与 VLA 思路相通
- [[Wan2.1|Wan2.2]]: VideoDiT 的预训练 backbone
- [[DiT|Diffusion Transformer]]: 共享主干架构

### 对比

- [[Pi05|π₀.₅]]: 上一代纯 action-only VLA 标杆
- [[Pi0|π₀]]: 原始 flow-matching VLA
- *Fast-WAM*: "每步都做梦"的 WAM 基线，AdaWAM 直接对标
- *Motus*: 另一条 video-action joint 路线
- *LingBot-VA*: 文本推理强但无视觉做梦
- *X-VLA*: 大规模动作-only VLA 基线
- *ACoT-VLA*: 链式思维 VLA，文本推理路径的对应物
- *GigaBrain-0*: 真机超大模型基线

### 方法相关

- [[Adaptive Multi-Modal Reasoning]]: 本文核心范式
- [[Dynamic Router]]: 路由器组件
- [[Text Reasoning Module]]: 文本推理路径
- [[Video-Action DiT]]: 共享生成主干
- [[Multimodal Reasoning Data Annotation]]: 自动标注 pipeline
- [[Action Chunking]]: 动作输出粒度
- [[Binary Cross-Entropy]]: 路由 loss
- [[VLM]]: 文本推理 backbone（Qwen3-VL）

### 硬件/数据相关

- [[LIBERO]]: 经典 4-suite VLA benchmark
- [[RoboTwin]]: 双手 manipulation benchmark
- 真机：双手机械臂平台（论文未指明型号）

---

## 速查卡片

> [!summary] AdaWAM (Dreaming when Necessary)
> - **核心**: 给 [[World-Action Model|WAM]] 装动态路由器，按 chunk 决定是否触发文本/视觉推理，实现 *"必要时才做梦"*
> - **方法**: Video-Action DiT 共享 [[Flow Matching]] 主干 + Qwen3-VL-4B 文本推理 + 轻量 MLP 路由器 + trajectory cue 自动标注 pipeline + 两阶段训练（50K + 10K 步）
> - **结果**: [[LIBERO]]-Long **99.1%** / [[RoboTwin]] 2.0 Hard **88.43%** / 真机 Clean Up Trash **70%** / Wipe Table **60%**; 推理延迟与 action-only Fast-WAM 持平; 组合泛化 unseen task **61%**（去 T.R. 仅 3%）
> - **代码**: https://adawam.github.io/ · arXiv 2606.07089

---

*笔记创建时间: 2026-06-09*
