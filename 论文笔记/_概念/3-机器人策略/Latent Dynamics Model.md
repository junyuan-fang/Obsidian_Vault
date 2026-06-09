---
type: concept
aliases: [潜在动力学模型, Latent Dynamics]
---

# Latent Dynamics Model

## 定义
在潜在空间（而非像素空间）中建模环境状态转移的动力学模型，将高维观测压缩到低维潜在表示后学习状态转移。

## 数学形式

$$
z_t = \text{Enc}(o_t), \quad \hat{z}_{t+1} = f_\theta(z_t, a_t), \quad \hat{o}_{t+1} = \text{Dec}(\hat{z}_{t+1})
$$

## 核心要点
1. 编码器将观测压缩为紧凑潜在表示，减少动力学建模的计算复杂度
2. 在潜在空间中进行状态转移预测，通常比像素空间预测更稳定
3. 是大多数现代世界模型（如 Dreamer 系列）的核心组件

## 代表工作
- [[DreamerV3]]: 使用 RSSM 作为潜在动力学模型
- [[Hi-WM]]: Visual Encoder → Latent Dynamics → Visual Decoder 架构

## 相关概念
- [[World Model]]
- [[Action-Conditioned World Model]]
