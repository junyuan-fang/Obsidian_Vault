---
type: concept
aliases: [Breadth-First Search, 广度优先搜索]
---

# BFS (Breadth-First Search)

## 定义

广度优先搜索，按层逐次扩展节点，在**无权图**或**单位代价图**上能找到最短路径。

## 在 STRIPS 规划中的角色

命题 STRIPS 规划默认每个动作代价为 1，BFS 在符号状态空间 $\{0,1\}^m$ 上自然给出最少动作的 plan。规模较小（$m \le 30$）时可直接用。

## 与本文关系

[[STRIPS-WM]] 测试时用 BFS 或 [[A* Search]] 作为经典规划器；baseline **WM-BFS** 则在 [[FSQ]] 抽象图上跑 BFS（无算子结构，泛化差）。

## 复杂度

- 时间: $O(b^d)$，$b$ 分支因子，$d$ 解深度
- 空间: $O(b^d)$（队列存所有未展开节点）

## 相关概念

- [[A* Search]]
- [[STRIPS]]
- [[Propositional Planning]]
