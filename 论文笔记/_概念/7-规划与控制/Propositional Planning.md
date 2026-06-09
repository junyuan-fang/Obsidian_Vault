---
type: concept
aliases: [命题规划, Propositional Task Planning]
---

# Propositional Planning

## 定义

经典 AI 规划的核心范式：世界状态用一组**布尔命题**（propositional fluents）的真值组合表示，动作通过谓词的增删改改变状态。最早形式化为 [[STRIPS]] (1971)，是 PSPACE-完全问题。

## 与 First-Order / Lifted 规划的区别

| 范式 | 状态表示 | 例子 |
|---|---|---|
| **Propositional / Grounded** | $m$ 个固定命题的 0/1 向量 | `on_A_B = 1` |
| **First-Order / Lifted** | 带变量的谓词 schema | `on(?x, ?y)` |

Grounded 版本针对具体物体集合，组合泛化弱；Lifted 版本可对任意物体推理，但搜索更难。

## 与本文关系

[[STRIPS-WM]] 学的是 **grounded propositional** STRIPS 模型——这是它的主要 limitation（论文 Section 7）：换一组新物体或新动作元数需要重训。

## 标准基准

- **BlocksWorld**: 经典玩具域，9 谓词
- **DinnerTable** ([[STRIPS-WM]] 提出)
- **IPC**（International Planning Competition）系列

## 相关概念

- [[STRIPS]]
- [[Propositional Predicate]]
- [[BFS]]
- [[A-star Search|A* Search]]
- [[Closed-World Assumption]]
