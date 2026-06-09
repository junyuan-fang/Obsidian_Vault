---
type: concept
aliases: [因果注意力, Causal Self-Attention, 因果掩码注意力]
---

# Causal Attention

## 定义
一种注意力机制约束，通过掩码确保每个 token 只能关注其之前（或特定集合内）的 token，防止信息泄露。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V
$$

其中 $M$ 为因果掩码矩阵，$M_{ij} = -\infty$ 当 token $i$ 不应关注 token $j$。

## 核心要点
1. GPT 系列模型中的标准做法，防止"看到未来"
2. 可扩展为块级因果掩码，对不同模态的 token 施加不同的注意力约束
3. GigaWorld-Policy 利用块级因果掩码实现动作预测与视频预测的解耦

## 代表工作
- [[GigaWorld-Policy]]: 块级因果掩码使动作 token 不依赖未来视频 token

## 相关概念
- [[Cross-Attention]]
- [[Diffusion Transformer]]
