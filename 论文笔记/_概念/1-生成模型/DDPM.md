---
type: concept
aliases: [Denoising Diffusion Probabilistic Model]
---

# DDPM

## 定义

Ho et al., 2020 提出的离散时间扩散生成模型：对数据 $x$ 加噪到 $\mathcal{N}(0,I)$，再训练网络反向去噪。

## 数学形式

前向：

$$
x_\tau = \sqrt{\bar\alpha_\tau}\, x_0 + \sqrt{1-\bar\alpha_\tau}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

训练目标（$\epsilon$-prediction）：

$$
\mathcal{L} = \mathbb{E}\bigl[\|\hat\epsilon_\theta(x_\tau, \tau) - \epsilon\|_2^2\bigr]
$$

## 核心要点

1. 离散 $\tau \in \{1, \dots, T\}$，典型 $T=1000$
2. $\bar\alpha_\tau$ 由 noise schedule（linear / cosine）决定
3. 反向用 $\hat\epsilon_\theta$ 迭代去噪，可加 DDIM 加速
4. **在策略学习中**：动作扩散（[[Diffusion Policy]]、[[NavDP]]）就是把 $x$ 换成动作序列

## 代表工作

- Ho et al., 2020: 起源
- DDIM / DPM-Solver: 加速采样
- [[Diffusion Policy]] / [[NavDP]]: 用于机器人动作

## 相关概念

- [[Diffusion Policy]]
- [[NavDP]]
