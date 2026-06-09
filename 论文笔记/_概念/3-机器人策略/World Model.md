---
type: concept
aliases: [世界模型, WM]
---

# World Model

## 定义
学习环境动力学的预测模型，给定当前状态和动作，预测未来状态的演化。在机器人领域用于想象式规划、合成数据生成和策略评估。

## 数学形式

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t), \quad \hat{o}_{t+1} = g_\phi(\hat{s}_{t+1})
$$

## 核心要点
1. 通常包含编码器（观测→潜在状态）、动力学模型（状态转移）、解码器（潜在状态→观测）
2. 可用于 model-based RL 的想象训练，减少真实交互需求
3. 近年在机器人操作领域用于策略评估和合成数据增强

## 代表工作
- [[DreamerV3]]: 基于世界模型的强化学习
- [[Hi-WM]]: 将世界模型用作人类纠正干预的交互式工作空间

## 相关概念
- [[Latent Dynamics Model]]
- [[Action-Conditioned World Model]]
