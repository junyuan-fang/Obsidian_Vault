---
type: concept
aliases: [VQ-VAE, VQ, 向量量化, vector-quantization]
---

# Vector Quantization (VQ-VAE)

## 定义

Vector Quantization Variational Auto-Encoder：把 encoder 输出的连续 latent 通过查询一个有限 codebook 量化为离散 token，下游 decoder 在离散 token 上重建。是 [[Latent Action Model|LAM]]、Cosmos Tokenizer、VQ-GAN 等离散表征学习的基础组件。

## 数学形式

$$
\mathcal{L}_{VQ} = \|x - \hat x\|^2 + \|\mathrm{sg}[z^q] - z\|^2 + \beta \|z^q - \mathrm{sg}[z]\|^2
$$

三项：重建 + codebook loss + commitment loss。$\mathrm{sg}[\cdot]$ 是 stop-gradient。

## 核心要点

1. **核心思想**: 在连续 latent 空间维护离散 codebook（通常 $|C| = 128 \sim 8192$），forward 时查最近邻 token
2. **梯度技巧**: straight-through estimator（前向用 $z^q$，反向用 $z$）
3. **commitment 损失**: $\beta$ 通常 0.25，鼓励 encoder 输出接近 codebook
4. **codebook collapse**: 容易出现部分 token 永不使用，需 EMA codebook 或 dead-code restart 缓解
5. **应用**: 离散视觉 token（VQ-GAN / [[Cosmos Tokenizer]]）、离散动作 token（[[Latent Action Model]] / LAPA）、音频 token（SoundStream）

## 代表工作

- [[Cosmos Tokenizer]]: NVIDIA 通用视频 tokenizer
- [[Latent Action Model|LAM]]: 离散化动作 latent
- [[LARA]]: 训练 LAM 时仍用 VQ，但对齐用 *量化前* 的连续 $z$

## 相关概念

- [[VAE]]
- [[Latent Action Model]]
- [[FSQ]]
