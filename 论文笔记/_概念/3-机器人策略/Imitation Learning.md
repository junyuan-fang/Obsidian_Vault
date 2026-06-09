---
type: concept
aliases: [模仿学习, IL, Learning from Demonstration, LfD]
---

# Imitation Learning

## 定义
从专家演示中学习策略的机器学习范式，目标是让智能体复现专家的行为模式，无需显式设计奖励函数。

## 数学形式

$$
\pi^* = \arg\min_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}} \left[ \mathcal{L}(\pi(s), a) \right]
$$

## 核心要点
1. 行为克隆（Behavioral Cloning）直接监督学习状态-动作映射
2. 面临分布偏移（distribution shift）问题：训练分布与测试分布不一致
3. DAgger 等方法通过在线数据聚合缓解分布偏移

## 代表工作
- [[Diffusion Policy]]: 基于扩散模型的模仿学习
- [[Hi-WM]]: 通过世界模型内的人类纠正扩展模仿学习数据

## 相关概念
- [[Human-in-the-Loop]]
- [[DAgger]]
- [[Diffusion Policy]]
