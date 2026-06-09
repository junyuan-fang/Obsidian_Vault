---
type: concept
aliases: [策略学坏, reward gaming, specification gaming]
---

# Reward Hacking

## 定义

策略学到了**在仿真器/奖励函数的漏洞下**得高分但**不解决真实任务**的捷径行为。在世界模型闭环训练中,策略可能利用世界模型的生成误差/物理不一致来"刷分"。

## 核心要点

1. 来源：奖励函数 / 仿真器存在可被钻空子的偏差
2. 与生成式世界模型搭配时风险更高（如 [[OmniDreams]] 局限性中提到）
3. 缓解：物理一致性约束、对抗仿真器、真车 sanity check

## 代表工作

- 大量 RL 安全文献
- [[OmniDreams]] 在批判性思考中明确提出此风险

## 相关概念

- [[Closed-Loop Simulation]]
- [[World-Action Model]]
