---
type: concept
aliases: [Causal Attention, 因果注意力]
---

# Causal Attention

## 定义
在 self-attention 上施加下三角 mask，使每个 token 只能 attend 到自身及历史 token，是自回归生成的基础。

## 数学形式
$$
$\text{Attn}_{causal}(Q,K,V)_i = \mathrm{softmax}((QK^\top)_{i,\le i})V_{\le i}$
$$

## 核心要点
1. GPT 风格 LM 的核心
2. 在视频/扩散里通常作用在时间维度，对帧内仍保持双向

## 代表工作
- [[minWM]]: Stage 1 把双向 DiT 改为因果生成

## 相关概念
- [[Self-Attention]]
- [[Causal Forcing]]
