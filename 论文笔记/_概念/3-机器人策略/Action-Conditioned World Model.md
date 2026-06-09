---
type: concept
aliases: [动作条件化世界模型, ACWM]
---

# Action-Conditioned World Model

## 定义
以动作为条件输入的世界模型，给定当前观测和执行的动作，预测下一时刻的观测或状态，支持闭环交互式推演。

## 数学形式

$$
\hat{o}_{t+1} = \mathcal{M}_\theta(o_t, a_t)
$$

## 核心要点
1. 与无条件预测（video prediction）不同，显式接收动作输入实现可控推演
2. 对动作维度和精度要求高，尤其在机器人操作场景中需处理高维连续动作
3. 可用于策略评估、想象训练、人机交互等多种下游任务

## 代表工作
- [[Hi-WM]]: 使用 14 维连续动作条件化的世界模型作为纠正工作空间

## 相关概念
- [[World Model]]
- [[Latent Dynamics Model]]
