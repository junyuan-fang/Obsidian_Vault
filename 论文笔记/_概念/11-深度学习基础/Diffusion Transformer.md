---
type: concept
aliases: [DiT, 扩散Transformer, Diffusion-Transformer]
---

# Diffusion Transformer

## 定义
将 Transformer 架构与扩散/流匹配生成框架结合的模型，用 Transformer 替代 U-Net 作为去噪网络的骨干，广泛用于图像和视频生成。

## 核心要点
1. 利用 Transformer 的全局注意力建模长程依赖
2. 支持多模态 token 的统一处理（视觉、语言、动作等）
3. 易于扩展到视频生成和机器人策略学习

## 代表工作
- [[GigaWorld-Policy]]: 5B 参数 Diffusion Transformer 作为视频生成骨干
- [[DiT]]: 原始 Diffusion Transformer 论文

## 相关概念
- [[Flow Matching]]
- [[VAE]]
- [[Causal Attention]]
