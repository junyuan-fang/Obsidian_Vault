---
type: concept
aliases: [Dataset Aggregation, 数据聚合]
---

# DAgger

## 定义
Dataset Aggregation 的缩写，一种交互式模仿学习算法，通过在学习者自身策略产生的状态上查询专家来缓解分布偏移问题。

## 数学形式

$$
\mathcal{D}_{i+1} = \mathcal{D}_i \cup \{(s, \pi^*(s)) : s \sim d_{\pi_i}\}
$$

## 核心要点
1. 核心思想：在策略自身访问的状态分布上收集专家标注
2. 每轮迭代中用当前策略 rollout，专家标注这些状态的最优动作
3. 理论上保证性能收敛到专家水平

## 代表工作
- Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (AISTATS 2011)
- [[Hi-WM]]: 将 DAgger 式纠正从物理世界转移到世界模型中

## 相关概念
- [[Imitation Learning]]
- [[Human-in-the-Loop]]
