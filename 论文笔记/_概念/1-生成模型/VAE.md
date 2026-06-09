---
type: concept
aliases: [变分自编码器, Variational Autoencoder]
---

# VAE

## 定义
变分自编码器（Variational Autoencoder），通过编码器将数据映射到隐空间、解码器从隐空间重建数据的生成模型，训练时优化 ELBO 下界。

## 数学形式

$$
\mathcal{L}_{VAE} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

## 核心要点
1. 编码器输出隐变量的均值和方差，通过重参数化技巧采样
2. 在扩散模型中常用于将图像编码为 latent 以降低计算成本（Latent Diffusion）
3. GigaWorld-Policy 使用预训练 VAE 将多视角图像编码为视觉 token

## 代表工作
- [[GigaWorld-Policy]]: 使用预训练 VAE 进行视觉 token 化

## 相关概念
- [[Diffusion Transformer]]
- [[Flow Matching]]
