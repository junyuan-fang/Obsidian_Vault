---
type: concept
aliases: [Separation Loss, Anti-Aliasing Loss]
---

# Predictive Separation Loss

## 定义

[[STRIPS-WM]] (Ajith & Chamzas 2026) 提出的辅助损失，用于减少 latent 表征的 **nondeterminism**——即"同一 $(z_t, a_t)$ 在数据集中对应多个不同 successor $z_{t+1}$"的情况。

## 触发条件与公式

当检测到两条样本 $(z^{(1)}_t, a^{(1)}_t, z^{(1)}_{t+1})$ 和 $(z^{(2)}_t, a^{(2)}_t, z^{(2)}_{t+1})$ 满足：

- $z^{(1)}_t = z^{(2)}_t$（量化后 code 相同）
- $a^{(1)}_t = a^{(2)}_t$（动作相同）
- $z^{(1)}_{t+1} \neq z^{(2)}_{t+1}$（后继不同）

则对两者的 **pre-quantization latent** $\tilde z^{(1)}_t, \tilde z^{(2)}_t$（连续值，未经 [[FSQ]] round）施加 separation 项，强迫它们距离拉开：

$$
\mathcal{L}_{\text{sep}} = -\| \tilde z^{(1)}_t - \tilde z^{(2)}_t \|
$$

下一次 round 时这两条样本就更有可能落入不同 FSQ 网格点，从而消除 latent nondeterminism。

## 作用

修复 [[Visual Aliasing]]——视觉上"看起来一样但 STRIPS 状态不同"的图像被错误归到同一 FSQ code。

## 相关概念

- [[FSQ]]
- [[JEPA]]
- [[Visual Aliasing]]
- [[Image-Grounded Task Graph]]
