---
type: concept
aliases: [LARA-alignment, latent-action-alignment]
---

# Latent Action Representation Alignment

## 定义

在 [[VLA]] 训练中，把 [[Latent Action Model|LAM]] 的 *在线* 连续 latent $z_t^\phi$ 与 [[VLA]] 主干（如 [[DiT]]）的中间层表征 $h_t^\theta$ 做 cosine 相似度对齐。是 [[Representation Alignment|REPA]] 思路从 image generation 到 VLA 的迁移，由 [[LARA]] 论文提出。

## 数学形式

$$
\mathcal{L}_{LARA}(\theta, \phi, \psi) = -\mathbb{E}\left[\mathrm{CosSim}\left(z_t^\phi, f_\psi(h_t^\theta)\right)\right]
$$

其中 $f_\psi$ 是 MLP projection head，$z_t^\phi$ 是在线 IDM 输出，$h_t^\theta$ 是 DiT 第 $L-2$ 层表征。

## 核心要点

1. **与传统 REPA 的差异**: teacher 不再 frozen，而是与 student 联合优化 → 双向梯度流
2. **与 pseudo-label 范式的差异**: 不把 LAM 离散 token 当监督，而对齐 *连续* latent → 信息损失更小
3. **双向正则化效应**:
   - VLA 的真实动作监督回流 → 逼 LAM 关注控制相关视觉变化
   - LAM 的视觉动力学先验 → 给 VLA 注入未来感知，减少功能不可达轨迹
4. **接入深度**: 一般在 DiT 的 $L-2$ 层（接近 action head 但保留语义）

## 代表工作

- [[LARA]]: 提出该范式，LIBERO-Long +12.4%、G1 真机 +32%

## 相关概念

- [[Representation Alignment]]
- [[Latent Action Model]]
- [[VLA]]
- [[DiT]]
