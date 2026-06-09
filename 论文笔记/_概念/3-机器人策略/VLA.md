---
type: concept
aliases: [Vision-Language-Action, 视觉-语言-动作模型]
---

# VLA

## 定义

Vision-Language-Action Model：把视觉观测 + 语言指令直接映射到机器人动作的多模态策略模型，是 [[VLM]] 在 Embodied AI 域的延伸,典型代表 [[Pi05|π0.5]]、[[Alpamayo]]、RT-2、OpenVLA。

## 数学形式

$$
\mathbf{u}_{t:t+H} = \pi_\theta(o_{t-K+1:t}, l)
$$

观测序列 + 语言指令 → [[Action Chunking|动作 chunk]]。

## 核心要点

1. **统一编码**: 视觉 patch token + 语言 token 拼接进 transformer，动作头输出连续/离散控制量
2. **训练数据**: 真机操作数据（DROID/BridgeV2/OXE）+ 仿真 + 视频预训练
3. **action head 形态**: AR 离散 / Diffusion / Flow Matching 均有
4. **泛化挑战**: 跨形态（embodiment）、跨场景、跨指令的零样本能力是核心评测

## 代表工作

- [[Alpamayo]]: NVIDIA 驾驶 VLA
- [[Pi05]] (π0.5): Physical Intelligence 通用 VLA
- [[Cosmos3]]: Cosmos3-Nano-Policy-DROID 配置作为 VLA 拿下 RoboArena 开源第一
- RT-2 / OpenVLA: 早期里程碑

## 相关概念

- [[VLM]]
- [[World-Action Model]]
- [[Action Chunking]]
- [[Diffusion Policy]]
