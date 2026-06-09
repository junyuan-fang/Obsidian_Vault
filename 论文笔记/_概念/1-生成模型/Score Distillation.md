---
type: concept
aliases: [Score Distillation, SDS, 分数蒸馏]
---

# Score Distillation

## 定义
用预训练扩散模型的 score 作为监督信号来训练另一个生成器（或参数化对象）的蒸馏家族。

## 数学形式
$$
$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon}[w(t)(\epsilon_\phi(x_t, t) - \epsilon)\partial x/\partial \theta]$
$$

## 核心要点
1. 代表：DreamFusion (SDS)、VSD、DMD、SiD
2. 核心思想：把教师 score 当作真实分布梯度的代理

## 代表工作
- [[minWM]]: DMD 是其家族成员，用于 Stage 3

## 相关概念
- [[DMD]]
- [[Score Identity Distillation]]
- [[Score Function]]
