---
type: concept
aliases: [Exposure Bias, 曝光偏差]
---

# Exposure Bias

## 定义
自回归模型在训练时看 ground-truth 历史、推理时看自己生成的历史，导致训练与推理分布不一致的问题。

## 核心要点
1. 典型出现在 seq2seq / 语言模型 / 自回归扩散
2. 缓解策略：scheduled sampling、DAgger、self-rollout、DMD 类分布匹配

## 代表工作
- [[minWM]]: Stage 3 的 self-rollout 即针对 exposure bias

## 相关概念
- [[Teacher Forcing]]
- [[Autoregressive Diffusion]]
