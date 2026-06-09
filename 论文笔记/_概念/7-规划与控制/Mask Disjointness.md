---
type: concept
aliases: [Add/Delete Disjointness, Well-Formed Operator]
---

# Mask Disjointness

## 定义

[[STRIPS]] 算子合法性的基本约束。对每个动作 $a$、每个谓词 $p$，需要：

$$
\text{add}_{a,p} + \text{del}_{a,p} \le 1
$$

$$
\text{pre}^+_{a,p} + \text{pre}^-_{a,p} \le 1
$$

即"加 + 删不能同时为 1"，"要求为真 + 要求为假不能同时为 1"。

## 直观含义

| 违反情形 | 不合理之处 |
|---|---|
| $\text{add} = \text{del} = 1$ | 同一动作同时让谓词为真又为假 |
| $\text{pre}^+ = \text{pre}^- = 1$ | 动作要求谓词既为真又为假，永不适用 |

## 与本文关系

[[STRIPS-WM]] 的 **Lemma 1** 证明：任何 CP-SAT 可行解都满足上述约束（通过 公式 8 显式编码），因此输出**良构 (well-formed) STRIPS 算子**。

## 相关概念

- [[STRIPS]]
- [[CP-SAT]]
- [[Propositional Predicate]]
