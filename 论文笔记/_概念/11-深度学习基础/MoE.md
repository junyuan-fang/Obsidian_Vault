---
type: concept
aliases: [Mixture of Experts, 混合专家模型]
---

# MoE

## 定义
Mixture of Experts 用多个专家子网络+门控网络，动态选择激活部分专家处理输入，实现参数扩展但计算成本可控。

## 数学形式
$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

## 核心要点
1. 门控网络决定 token 路由到哪些专家
2. 总参数量大但每 token 只激活少量专家
3. 负载均衡是关键挑战

## 相关概念
- [[Transformer]]