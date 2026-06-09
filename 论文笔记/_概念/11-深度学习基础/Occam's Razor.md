---
type: concept
aliases: [奥卡姆剃刀, Parsimony Principle, Minimum Description Length]
---

# Occam's Razor

## 定义

哲学/机器学习的简洁性原则：**在解释相同数据的多个假设中，选择最简单的那个**。在统计学习理论里对应**最小描述长度 (MDL)** 与稀疏正则化。

## 在 STRIPS 学习中的体现

[[STRIPS-WM]] 的字典序目标函数（公式 1）的后两层：

$$
\min \sum_{a,p} (\text{add}_{a,p} + \text{del}_{a,p}), \quad \min \sum_{a,p} (\text{pre}^+_{a,p} + \text{pre}^-_{a,p})
$$

即在保证正确性的前提下，**让 add/delete mask 与 precondition 都尽量稀疏**。这正是 Occam's Razor 在算子学习中的实例化——简洁的算子更可能反映真实任务结构而非偶然拟合。

## 为什么重要

- 防止过拟合到偶然出现的转移噪声
- 提升可解释性
- 提升组合泛化能力（不必要的 precondition 会阻塞合法 plan）

## 相关概念

- [[STRIPS]]
- [[CP-SAT]]
