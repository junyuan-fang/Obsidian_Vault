---
type: concept
aliases: [关键点对齐损失]
---

# Keypoint Alignment Loss

## 定义
把 [[SMPL-X]] 关键点投影到 2D 与图像检测得到的 2D 关键点之间的距离损失，是 HOI / 人体重建的标准能量项。

## 数学形式
$$
\mathcal{L}_{kp} = \frac{1}{T}\sum_{t=1}^{T} \left\|\mathcal{K}^H(\Theta^H_t) - p_t\right\|
$$

## 核心要点
1. **L1 范式**：对异常点更鲁棒；
2. **置信度加权**（变体）：用 detection confidence 加权。

## 代表工作
- [[GRAIL]]

## 相关概念
- [[GRAIL Reconstruction Loss]]
- [[SMPL-X]]
