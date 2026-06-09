---
type: concept
aliases: [Proximal Policy Optimization, 近端策略优化]
---

# PPO

## 定义
一种 on-policy 策略梯度算法，通过 clipping 限制策略更新幅度，兼顾训练稳定性和样本效率。

## 数学形式
$$L^{CLIP}(	heta) = \mathbb{E}_t[\min(r_t(	heta)\hat{A}_t, 	ext{clip}(r_t(	heta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

## 核心要点
1. 比 TRPO 实现简单，效果相当
2. 适用于连续和离散动作空间
3. 是 humanoid locomotion RL 训练的默认选择

## 相关概念
- [[DreamerV3]]