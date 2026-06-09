---
type: concept
aliases: [Variational Autoencoder, 变分自编码器]
---

# VAE (Variational Autoencoder)

## 定义

把数据通过编码器映射到低维隐变量分布 $q_\phi(z\mid x)$，再用解码器 $p_\theta(x\mid z)$ 重建的概率生成模型，训练目标为最大化 ELBO（重建 + [[KL Divergence]] 正则）。

## 数学形式

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log p_\theta(x\mid z)\right] - \mathrm{KL}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)
$$

## 核心要点

1. **隐空间连续**：与 [[FSQ]]/VQ-VAE 离散 token 不同，VAE latent 是连续高斯
2. **现代主流用法**：图像/视频 [[Diffusion Model|diffusion]] 几乎全在 VAE latent space 训练（SD/Cosmos/MMDiT），称 *latent diffusion*
3. **机器人/WM 用法**：用 VAE feature 当作"压缩视觉表示"，World Expert 只预测 feature 不预测像素（如 [[WLA]] 的隐式世界建模）
4. **缺点**：解码可能模糊，重建质量受限于 latent 维度

## 代表工作

- 经典：Kingma & Welling, *Auto-Encoding Variational Bayes* (2013)
- Latent Diffusion (Stable Diffusion) 系列
- [[WLA]]: 用 VAE feature 当 World Expert 预测目标

## 相关概念

- [[KL Divergence]]
- [[Diffusion Model]]
- [[FSQ]]
- [[Cosmos Tokenizer]]
