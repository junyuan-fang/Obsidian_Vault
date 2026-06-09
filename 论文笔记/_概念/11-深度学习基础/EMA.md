---
type: concept
aliases: [Exponential Moving Average, Momentum Encoder, EMA Teacher]
---

# EMA (Exponential Moving Average)

## 定义

对一组参数 $\theta$ 维护一份滑动平均副本 $\bar\theta$：

$$
\bar\theta \leftarrow \mu \bar\theta + (1 - \mu) \theta
$$

其中 $\mu$ 是动量（典型值 0.99–0.9999）。

## 在自监督学习中的用途

构造**稳定的 teacher 目标**：student 参数 $\theta$ 用梯度更新，teacher 参数 $\bar\theta$ 用 EMA 更新（**不接梯度**）。这种 student-teacher 解耦避免了表征塌缩：

- BYOL: $\mu = 0.99 \to 0.9999$ schedule
- MoCo: $\mu = 0.999$
- DINO: $\mu = 0.996 \to 1.0$
- [[JEPA]] / I-JEPA: $\mu \approx 0.996$
- [[STRIPS-WM]]: $\mu = 0.996$

## 在模型平均中的另一种用途

训练完整后用 EMA 权重替换最终模型（如 Stable Diffusion、扩散模型训练），通常涨点 ~1%。

## 相关概念

- [[JEPA]]
- [[Teacher Forcing]]
