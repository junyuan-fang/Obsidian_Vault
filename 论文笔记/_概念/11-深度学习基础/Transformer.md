---
type: concept
aliases: [Transformer 架构]
---

# Transformer

## 定义
基于自注意力机制的序列建模架构，通过并行计算替代 RNN 的串行处理，成为现代深度学习的基础架构。

## 核心要点
1. 由 Encoder 和 Decoder 组成，核心是 Multi-Head Self-Attention
2. 位置编码（Positional Encoding）补充序列顺序信息
3. 已扩展到视觉（ViT）、视频生成（DiT）等领域

## 代表工作
- [[DiT]]: Transformer-based 扩散模型
- [[MultiWorld]]: 基于 Transformer backbone 的视频世界模型

## 相关概念
- [[Attention]]
- [[RoPE]]
- [[DiT]]
