---
type: concept
aliases: [Adam with Weight Decay, AdamW优化器]
---

# AdamW

## 定义
一种改进的 Adam 优化器，将权重衰减与梯度更新解耦，避免 L2 正则化在自适应学习率下的失效问题。

## 数学形式

$$
\theta_{t+1} = \theta_t - \eta \Big(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\Big)
$$

## 核心要点
1. 权重衰减直接作用于参数，而非通过损失函数间接作用
2. 在 Transformer 训练中广泛使用
3. 通常配合余弦学习率调度

## 代表工作
- [[GigaWorld-Policy]]: 使用 AdamW ($\beta_1=0.85, \beta_2=0.9$) 训练

## 相关概念
- [[Diffusion Transformer]]
