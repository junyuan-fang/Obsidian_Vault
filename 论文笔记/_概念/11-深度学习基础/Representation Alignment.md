---
type: concept
aliases: [REPA, representation-alignment, 表征对齐]
---

# Representation Alignment (REPA)

## 定义

Representation Alignment：在生成模型 / 策略模型训练时，额外加一条 loss 把模型 *中间层表征* 对齐到一个外部 *已经学好的* 表征空间（如 DINO、MAE 的 visual feature，或在线训练的 [[Latent Action Model|LAM]] latent），从而注入先验、加速收敛、提升质量。

## 数学形式

$$
\mathcal{L}_{RA}(\theta, \psi) = -\mathbb{E}\left[\mathrm{CosSim}\left(y^\text{teacher}, f_\psi(h^\theta)\right)\right]
$$

通常用 cosine 相似度，加可学习 MLP projection $f_\psi$ 适配维度。

## 核心要点

1. **起源**: Yu et al. 2024 "Representation Alignment for Generation" → 加速 [[DiT]] 收敛
2. **关键设计点**:
   - 接入深度：通常中间层（不要太浅也不要太接近输出）
   - teacher 选择：frozen DINO/MAE 是标准 baseline
   - projection head：MLP 防维度 / 表征空间错配
3. **在 VLA 的扩展**: [[Latent Action Representation Alignment|LARA]] 把 teacher 从 frozen 改成 *online* LAM，构成双向梯度流
4. **直觉**: 让生成模型 / 策略模型 "提前学到" teacher 表征空间的几何，少绕远路

## 代表工作

- REPA (Yu et al., 2024)：扩散模型加速训练
- [[LARA]]：迁移到 VLA + 在线 teacher

## 相关概念

- [[Latent Action Representation Alignment]]
- [[DiT]]
- [[VLA]]
