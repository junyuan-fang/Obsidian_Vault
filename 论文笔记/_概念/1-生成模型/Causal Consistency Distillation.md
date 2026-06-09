---
type: concept
aliases: [Causal Consistency Distillation, CCD, 因果一致性蒸馏]
---

# Causal Consistency Distillation

## 定义
在自回归扩散学生上做 Consistency Distillation：要求生成器在相邻 timestep 上自洽，无需 offline ODE 数据。

## 数学形式
$$
$\min_\theta \mathbb{E}[w(t)\,d(G_\theta(x_t^i, x_{gt}^{<i}, t),\,G_{\theta^-}(\hat x_{t-\Delta t}^i, x_{gt}^{<i}, t-\Delta t))]$
$$

## 核心要点
1. $\theta^-$ 是 EMA 教师
2. 不依赖预先采样的 ODE 轨迹，节省存储
3. 是 Consistency Distillation 的 causal 版本

## 代表工作
- [[minWM]]: Stage 2b

## 相关概念
- [[Causal Forcing]]
- [[Score Distillation]]
