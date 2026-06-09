---
type: concept
aliases: [Mixture of Experts, 混合专家]
---

# MoE

## 定义

Mixture-of-Experts：把 FFN 层拆成多个并行专家，每个 token 通过 gating 网络选 Top-k 专家激活，参数总量大但激活计算量恒定，是 LLM 稀疏放大的主流路线。

## 数学形式

$$
y = \sum_{k \in \text{Top-}k(\,G(x))} G(x)_k \cdot E_k(x)
$$

其中 $G(x)$ 是 gating 分数，$E_k$ 是第 $k$ 个专家 FFN。

## 核心要点

1. **稀疏激活**: 仅 Top-k 个专家参与计算，FLOPs 不随专家数线性增长
2. **路由难题**: 易出现专家坍缩、负载不均，需 load-balancing loss
3. **训练不稳**: gating 信号是离散选择，反传需 straight-through 或可微近似
4. 与 [[Mixture-of-Transformers|MoT]] 不同：MoE 按 token learned gating，MoT 按模态确定性路由

## 代表工作

- Switch Transformer / GShard / Mixtral：经典 LLM MoE
- DeepSeek-MoE、Qwen3-MoE：中文 LLM 的稀疏路线
- [[Cosmos3]] 的 [[Mixture-of-Transformers|MoT]] 借鉴 MoE 但简化路由

## 相关概念

- [[Mixture-of-Transformers]]
- [[Self-Attention]]
