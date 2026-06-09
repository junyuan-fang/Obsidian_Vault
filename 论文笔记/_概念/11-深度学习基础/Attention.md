---
type: concept
aliases: [注意力机制, Self-Attention, Scaled Dot-Product Attention]
---

# Attention

## 定义
一种让模型动态聚焦于输入中相关部分的机制，通过 Query-Key-Value 计算加权求和。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 核心要点
1. Scaled Dot-Product Attention 通过 $\sqrt{d_k}$ 缩放防止梯度消失
2. Multi-Head Attention 并行计算多组注意力以捕获不同子空间信息
3. 是 Transformer 架构的核心计算单元

## 代表工作
- [[Transformer]]: 首次提出 self-attention 机制

## 相关概念
- [[Cross-Attention]]
- [[Transformer]]
- [[RoPE]]
