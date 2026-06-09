---
type: concept
aliases: [交叉注意力, Cross Attention]
---

# Cross-Attention

## 定义
一种注意力变体，Query 来自一个序列，Key 和 Value 来自另一个序列，实现跨模态/跨来源的信息融合。

## 数学形式

$$
\text{CrossAttn}(Q_x, K_y, V_y) = \text{softmax}\left(\frac{Q_x K_y^T}{\sqrt{d_k}}\right)V_y
$$

## 核心要点
1. 广泛用于条件生成（如文本条件图像生成）
2. Q 来自主序列，KV 来自条件序列
3. 在 DiT 中常用于注入文本/动作条件信息

## 代表工作
- [[MultiWorld]]: 用 Cross-Attention 将 VGGT 全局状态注入生成过程

## 相关概念
- [[Attention]]
- [[Transformer]]
