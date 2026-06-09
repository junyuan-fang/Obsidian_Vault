---
type: concept
aliases: [Score Identity Distillation, SiD]
---

# Score Identity Distillation

## 定义
Score Distillation 的一个变体，使用恒等映射目标使学生一步生成与扩散教师分布匹配，理论上无需额外 critic。

## 数学形式
$$
$\nabla_\theta \mathcal{L}_{SiD} \approx \mathbb{E}[(s_{real}(x_g) - s_{real}(\bar x_g))\,\partial x_g/\partial \theta]$
$$

## 核心要点
1. 相比 DMD 不需维护 fake score 网络
2. 可把 NFE 进一步压到 1 步

## 代表工作
- [[minWM]]: 讨论的潜在改进方向

## 相关概念
- [[DMD]]
- [[Score Distillation]]
