---
type: concept
aliases: [WM, 世界模型]
---

# World Model

## 定义

学习环境动力学的生成模型：给定历史观测与候选动作，预测未来观测/状态，用于 model-based RL / 想象式规划 / 数据扩充。

## 核心要点

1. **典型形式**：VAE-RNN（Ha & Schmidhuber 原版）、Transformer、Diffusion
2. **应用**：DreamerV3 系列 / Genie / NVIDIA Cosmos / 自动驾驶世界模型
3. **在 VLN 中潜力**：可作为 oracle 替代 A\*，给任意状态生成"如果继续走会怎样"的标签

## 代表工作

- Ha & Schmidhuber: 原版 World Models
- DreamerV3, Genie, Cosmos
- 在 [[GN0]] 的批判性分析中：用 WM 替代 A\* DAgger oracle 是潜在改进方向

## 相关概念

- [[Vision-Language Navigation]]
