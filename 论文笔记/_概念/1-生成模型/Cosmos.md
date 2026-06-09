---
type: concept
aliases: [Cosmos World Foundation Model, NVIDIA Cosmos]
---

# Cosmos

## 定义

NVIDIA 推出的开源 [[Video Diffusion Model|视频世界基础模型]] 系列，在 2000 万小时通用视频上预训练，作为 [[Embodied AI|具身智能]] / 自动驾驶 / 机器人的"世界视觉先验"。

## 数学形式

通用条件视频生成训练目标：

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}, c}\!\left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) \|^2 \right]
$$

其中 $c$ 可为文本、图像、动作等多模态条件。

## 核心要点

1. 开源 + 多版本（Predict、Reason、Cosmos 3 Nano/Super 等）
2. 同时提供扩散与自回归两套主干
3. 设计为可被下游领域（驾驶、机器人）做 mid-training / post-training 的底座
4. 已成为 NVIDIA 物理 AI 生态（Alpamayo、OmniDreams、AlpaGym 等）的共同基础

## 代表工作

- [[Cosmos3]]: 五模态 omnimodal world model,把 Cosmos 系列升级为统一 Physical AI backbone（Nano 16B / Super 65B）
- [[OmniDreams]]: 在 Cosmos 基础上后训练 21,000 小时驾驶视频，做闭环自动驾驶世界模型

## 相关概念

- [[Video Diffusion Model]]
- [[Diffusion Model]]
- [[World-Action Model]]
