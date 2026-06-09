---
type: concept
aliases: [Dreamer V3]
---

# DreamerV3

## 定义
基于世界模型的 model-based RL 算法，在学到的 latent world model 中做 imagination rollout 训练策略。

## 核心要点
1. Symlog 预测 + KL 正则化 + free bits
2. 固定超参跨任务通用
3. 在 Atari/DMC/Minecraft 等任务上 SOTA

## 相关概念
- [[PPO]]
- [[JEPA]]