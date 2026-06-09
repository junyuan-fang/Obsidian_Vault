---
type: concept
aliases: [Self-Attention, 自注意力]
---

# Self-Attention

## 定义
Transformer 的核心模块，用同一序列计算 Q/K/V，输出每个 token 对所有其他 token 加权聚合的结果。

## 数学形式
$$
$\text{Attn}(Q,K,V) = \mathrm{softmax}(QK^\top / \sqrt{d})V$
$$

## 核心要点
1. 复杂度 $O(N^2 d)$
2. 可通过 mask 实现 causal / window / sparse
3. RoPE/PRoPE 等位置编码通常作用在 Q/K 上

## 代表工作
- [[minWM]]: PRoPE 在 self-attention 的 Q/K/V 上做几何投影

## 相关概念
- [[RoPE]]
- [[Causal Attention]]
- [[MMDiT]]
