---
type: concept
aliases: [Score Function, 得分函数, 分数函数]
---

# Score Function

## 定义
概率密度对数关于输入的梯度 $\nabla_x \log p(x)$，是扩散模型反向过程的核心量。

## 数学形式
$$
$s(x) = \nabla_x \log p(x)$
$$

## 核心要点
1. 扩散模型本质是学习时间相关的 score $s_\theta(x, t)$
2. Tweedie 公式连接 score 与去噪：$\mathbb{E}[x_0|x_t] = x_t + \sigma_t^2 s(x_t)$

## 代表工作
- [[minWM]]: Stage 3 的 DMD 用真假 score 差作为梯度

## 相关概念
- [[Diffusion Model]]
- [[DMD]]
- [[Score Distillation]]
