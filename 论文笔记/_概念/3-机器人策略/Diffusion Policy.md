---
type: concept
aliases: [扩散策略, DP]
---

# Diffusion Policy

## 定义
将扩散生成模型应用于机器人动作生成的策略学习框架，通过迭代去噪过程从噪声中生成多模态动作分布。

## 数学形式

$$
a_{0:K} = \text{Denoise}(a_T, o_t; \theta), \quad a_T \sim \mathcal{N}(0, I)
$$

## 核心要点
1. 将动作生成建模为条件去噪扩散过程，天然支持多模态动作分布
2. 使用 DDPM/DDIM 采样生成动作序列（action chunk）
3. 支持 CNN-based 和 Transformer-based 两种架构变体

## 代表工作
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
- [[Hi-WM]]: 作为策略骨干之一

## 相关概念
- [[Imitation Learning]]
- [[Action Space]]
