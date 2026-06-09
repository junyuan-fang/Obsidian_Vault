---
type: concept
aliases: [命题谓词, Propositional Fluent, Boolean Fluent]
---

# Propositional Predicate

## 定义

经典符号 AI 中描述世界状态的最基本单位——一个**布尔值**事实，例如 `block_A_on_B = 1` 或 `gripper_holding = 0`。

整个世界状态用一组 $m$ 个谓词的 0/1 向量 $x \in \{0,1\}^m$ 表示。

## 命名 vs 含义

谓词只有"名字 + 0/1 值"，**没有内置语义**——一个学习到的谓词向量第 $p$ 维代表什么，由对应动作的 precondition/effect mask 间接定义。这是 [[STRIPS-WM]] 等神经符号方法的关键设计：**谓词被自动发现，不需要人工命名**。

## 与本文关系

[[STRIPS-WM]] 通过 [[CP-SAT]] 同时求解每个状态的谓词向量 $x_{\hat s} \in \{0,1\}^m$ 与每个动作的 mask，让"谓词到底是什么"由数据自动决定。论文中 $m$ 取 9–35。

## 相关概念

- [[STRIPS]]
- [[Propositional Planning]]
- [[Mask Disjointness]]
