---
type: concept
aliases: [MoT, 模态混合 Transformer]
---

# Mixture-of-Transformers

## 定义

一种统一多模态架构：Transformer 中 [[Self-Attention|attention]] 层跨模态共享，FFN 层按模态路由到专属专家（类比 [[MoE]] 但路由按模态确定而非 learned gating），兼顾跨模态对齐与模态特化。

## 数学形式

$$
h^{(\ell)}_{m,i} = \text{Attn}^{(\ell)}\!\left(\{h^{(\ell-1)}_{m',j}\}_{\forall m',j}\right) + \text{FFN}^{(\ell)}_{m}\!\left(h^{(\ell-1)}_{m,i}\right)
$$

其中 $m$ 为模态标签（语言/图像/视频/音频/动作等）。

## 核心要点

1. 共享 attention 让跨模态 token 直接交互，保留对齐能力
2. 模态专属 FFN 避免参数冲突，类似 MoE 但**路由确定性**，训练更稳
3. 与 Janus / Chameleon 类完全共享参数的方案相比，模态特定容量更可控
4. 与 [[MoE]] 区别：MoE 按 token gating 选 expert，MoT 按模态先验路由

## 代表工作

- [[Cosmos3]]: 五模态 omnimodal world model 的统一 backbone
- 前序工作: Mixture-of-Transformers (Liang et al., 2024) 提出该结构

## 相关概念

- [[MoE]]
- [[MMDiT]]
- [[Self-Attention]]
