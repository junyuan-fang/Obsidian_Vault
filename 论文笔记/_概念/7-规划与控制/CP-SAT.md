---
type: concept
aliases: [Constraint Programming SAT, OR-Tools CP-SAT]
---

# CP-SAT

## 定义

Google OR-Tools 中的**约束规划 + SAT 求解器**，是 lazy clause generation 风格的现代 CP 求解器。它把约束规划问题转化为 SAT 形式，由强大的 SAT engine 加 CP 风格的传播器（cumulative, table, allDifferent 等）共同求解。

## 能力范围

- 布尔变量 + 整数变量
- 线性约束 / 表约束 / 累积约束
- 目标函数（包括字典序最小化 `min_lex`）
- 时间预算 + 多解枚举

## 在 STRIPS 学习中的角色

[[STRIPS-WM]] 把"找一组谓词 + 算子使所有观察转移可解释"建模为 CP-SAT：

- **变量**: 谓词向量 $x_{\hat s, p}$、precondition / add / delete mask、松弛 $\xi, \eta$，全部 0/1
- **约束**: 公式 4–11（C1 前置、C2 效果、C3 负证据、Mask 互斥）
- **目标**: 字典序四层（公式 1）
- **预算**: 300 s per fixed-$m$

## 为什么选 CP-SAT

| 替代方案 | 缺点 |
|---|---|
| MILP | 字典序目标麻烦，0/1 mask 表达冗长 |
| 端到端 NN | 无可证明性，长 horizon 退化 |
| 纯 SAT | 没有内置整数 / 字典序支持 |

## 相关概念

- [[STRIPS]]
- [[Mask Disjointness]]
- [[STRIPS-WM]]
