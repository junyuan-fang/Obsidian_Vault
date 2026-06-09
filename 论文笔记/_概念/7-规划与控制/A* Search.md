---
type: concept
aliases: [A-star, A*, A star Search]
---

# A* Search

## 定义

经典最优启发式搜索算法，由 Hart, Nilsson & Raphael (1968) 提出。在带权图上寻找从起点到目标的**最短路径**，每次扩展时按 $f(n) = g(n) + h(n)$ 排序：

- $g(n)$: 从起点到 $n$ 的已知最低代价
- $h(n)$: 从 $n$ 到目标的**启发估计**（admissible heuristic 时保证最优）

## 在 STRIPS 规划中的角色

在命题状态空间 $\{0,1\}^m$ 上展开后继时，A\* 比纯 [[BFS]] 高效得多。常用启发式：

- $h^{\max}$: 达到任一目标谓词的最大单步代价
- $h^+$: delete-relaxation 启发式
- LM-cut, $h^{\text{FF}}$（FastForward）

## 与本文关系

[[STRIPS-WM]] 学到符号模型后，测试时直接用 BFS / A\* 在 $m$ 维谓词空间搜索。

## 代表工作

- Hart et al. 1968: 原始 A\*
- FastForward, LAMA: 现代 STRIPS heuristic planner
- [[STRIPS-WM]]: 学符号后用 A\* 推理

## 相关概念

- [[BFS]]
- [[STRIPS]]
- [[Propositional Planning]]
