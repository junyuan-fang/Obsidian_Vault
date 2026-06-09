---
type: concept
aliases: [Decoupled Clip and Dynamic Sampling Policy Optimization]
---

# DAPO

## 定义

Yu et al., 2025 提出的 LLM RL 优化框架，基于 [[GRPO]] 的组归一化优势 + 解耦的非对称 ratio clipping，用于大模型的策略后训练。

## 数学形式

$$
\mathcal{J}_{DAPO}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\bigl(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon_{low}, 1+\epsilon_{high}) A_i\bigr)
$$

## 核心要点

1. **解耦 clip**：上下界 $\epsilon_{low}, \epsilon_{high}$ 不对称，允许更激进的正向更新
2. **GRPO 优势**：无 value head，组内 z-score 归一化
3. **可关闭 KL**：在格式约束已由 reward 显式惩罚保证的场景，关闭 KL 可避免破坏结构化输出
4. **应用场景**：长 chain-of-thought / 结构化输出 / 智能体后训练

## 代表工作

- [[GN0]]: 用 DAPO 对 [[DAgger]] 初始化后的策略做轨迹级 RL 精炼
- DeepSeek-R1 系列后训练

## 相关概念

- [[GRPO]]
- [[DAgger]]
- [[SFT]]
