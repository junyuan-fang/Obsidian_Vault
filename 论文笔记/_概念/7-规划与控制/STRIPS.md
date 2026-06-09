---
type: concept
aliases: [Stanford Research Institute Problem Solver, STRIPS Planning]
---

# STRIPS (Stanford Research Institute Problem Solver)

## 定义

经典符号 AI 规划范式，由 Fikes & Nilsson (1971) 提出。世界状态用一组**命题谓词**（propositional fluents）的真值组合描述，动作用三个 mask 描述：

- **Precondition**（前置条件）：动作 $a$ 可执行需要满足的谓词
- **Add list**（增加效果）：执行后变为真的谓词
- **Delete list**（删除效果）：执行后变为假的谓词

## 数学形式

状态 $\sigma \in \{0,1\}^m$ 是 $m$ 个谓词的 0/1 向量。动作 $a$ 由四元组定义：

$$
a = \langle \text{pre}^+_a,\ \text{pre}^-_a,\ \text{add}_a,\ \text{del}_a \rangle
$$

转移函数（命题语义）：

$$
\sigma' = \tau_a(\sigma) = \text{add}_a \vee (\sigma \wedge \neg \text{del}_a)
$$

适用性条件：$\text{pre}^+_a \le \sigma \ \wedge\ \text{pre}^-_a \le 1 - \sigma$（按位）。

## 核心要点

1. **可计算性**: 命题 STRIPS 规划是 PSPACE-完全
2. **完备规划器**: BFS / [[A* Search]] / heuristic search（如 FF, LAMA）可在符号空间高效搜索
3. **与现代 ML 的桥梁**: [[LatPlan]] / [[STRIPS-WM]] 试图从图像数据中自动学习 STRIPS 模型
4. **Mask 互斥**: 良构 STRIPS 要求 $\text{add}_a \cap \text{del}_a = \emptyset$ 且 $\text{pre}^+_a \cap \text{pre}^-_a = \emptyset$

## 代表工作

- Fikes & Nilsson 1971: 原始 STRIPS
- [[LatPlan]] (Asai et al. 2022): 神经网络学 STRIPS
- [[STRIPS-WM]] (Ajith & Chamzas 2026): CP-SAT 求解 STRIPS 算子学习

## 相关概念

- [[Propositional Planning]]
- [[Propositional Predicate]]
- [[A* Search]]
- [[BFS]]
- [[Closed-World Assumption]]
- [[Mask Disjointness]]
- [[CP-SAT]]
