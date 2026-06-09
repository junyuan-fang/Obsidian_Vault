---
type: concept
aliases: [Causal Forcing, 因果强制]
---

# Causal Forcing

## 定义
把双向扩散模型转为自回归生成的蒸馏框架：用 causal attention mask + teacher forcing 训练学生只看历史。

## 数学形式
$$
$\mathcal{L} = \mathbb{E}\|G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i\|^2$
$$

## 核心要点
1. Stage 1：causal mask + teacher forcing
2. Stage 2：少步生成器初始化（Causal ODE）
3. Stage 3：分布对齐（DMD-style）
4. Causal Forcing++ 加入 self-rollout 以减少 exposure bias

## 代表工作
- [[minWM]]: 整体蒸馏框架

## 相关概念
- [[Causal Consistency Distillation]]
- [[Causal ODE Initialization]]
- [[Teacher Forcing]]
