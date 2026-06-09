---
type: concept
aliases: [CWA, 闭世界假设]
---

# Closed-World Assumption (CWA)

## 定义

知识表示中的一种推理假设：**未被陈述为真的事实，被默认为假**。与"开世界假设" (OWA) 相对——OWA 把未陈述的事实视为"未知"。

## 在 STRIPS 规划中

经典 STRIPS 规划默认 CWA：每条 ground predicate 不在初始状态中就视为 0。这保证了状态可以用有限 0/1 向量精确表示。

## 在 STRIPS 学习中的 Local CWA

[[STRIPS-WM]] 不能对整个数据集做全局 CWA（很多状态-动作未充分采样，"没观察"不等于"不可行"）。所以引入 **trusted missing pair**：只对出度饱和的状态做局部 CWA，仅在那里把"未观察"当作"动作不可行"。这是 [[Trusted Missing Pair]] 概念的本质。

## 与本文关系

[[STRIPS-WM]] 学习 precondition 时的负证据机制建立在 local CWA 上。

## 相关概念

- [[Trusted Missing Pair]]
- [[STRIPS]]
- [[Propositional Planning]]
