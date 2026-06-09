---
type: concept
aliases: [OmniDreams 自动驾驶世界模型]
---

# OmniDreams

## 定义

NVIDIA Research 提出的实时生成式自动驾驶世界模型：在 [[Cosmos]] [[Video Diffusion Model|视频扩散]] 先验上后训练 21,000 小时驾驶数据，做动作条件、自回归、实时闭环传感器视频生成,作为 [[Closed-Loop Simulation|闭环仿真]] 的"神经世界"。

## 核心要点

1. 基于 [[Cosmos]] mid+post training 而非从零训
2. [[World-Action Model|WAM]] 作为策略 backbone 替代 [[VLA|VLA]]
3. 与 Alpamayo 1、AlpaSim 协调器组成完整闭环训练栈
4. [[DMD|蒸馏]] 到 few-step 实现实时推理

## 代表工作

- [[OmniDreams]] 论文（arXiv:2606.03159, NVIDIA 2026）

## 相关概念

- [[Cosmos]]
- [[Cosmos3]]
- [[World-Action Model]]
- [[Alpamayo]]
- [[Causal Forcing]]
