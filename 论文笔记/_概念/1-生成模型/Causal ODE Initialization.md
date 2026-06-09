---
type: concept
aliases: [Causal ODE Initialization, Causal ODE init]
---

# Causal ODE Initialization

## 定义
把少步自回归生成器初始化为从噪声中间帧到 clean 帧的回归，作为 Consistency Distillation 的稳定起点。

## 数学形式
$$
$\theta^\star = \arg\min_\theta \mathbb{E}\|G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i\|^2$
$$

## 核心要点
1. 本质是回归型 ODE 反演的近似
2. 若跳过此步直接做 Consistency，往往发散

## 代表工作
- [[minWM]]: Stage 2a

## 相关概念
- [[Causal Forcing]]
- [[Causal Consistency Distillation]]
