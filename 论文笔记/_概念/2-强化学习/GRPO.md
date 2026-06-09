---
type: concept
aliases: [Group Relative Policy Optimization]
---

# GRPO

## 定义

Shao et al., 2024 (DeepSeekMath) 提出的策略梯度变体：每条 prompt 采样一组 $G$ 个响应，组内 z-score 归一化得到相对优势，免去独立 critic/value 网络。

## 数学形式

$$
A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

配合 PPO 风格 clipped policy gradient 使用。

## 核心要点

1. **无 critic**：省去 value head 训练成本，避免 value 估计不准
2. **组内比较**：把"全局奖励 scaling"变成"同 prompt 内相对优劣"，更稳
3. **PPO 兼容**：直接代入 PPO 的 ratio clipping 框架
4. **典型 $G$**：8 ~ 16

## 代表工作

- DeepSeekMath / DeepSeek-R1: 起源
- [[DAPO]]: 基于 GRPO 的解耦 clip 变体
- [[GN0]]: 在导航策略后训练中用 DAPO/GRPO

## 相关概念

- [[DAPO]]
- [[SFT]]
- [[DAgger]]
