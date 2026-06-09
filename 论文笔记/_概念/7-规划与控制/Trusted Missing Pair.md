---
type: concept
aliases: [TMP, Negative Evidence Pair]
---

# Trusted Missing Pair

## 定义

[[STRIPS-WM]] (Ajith & Chamzas 2026) 提出的概念。对一个抽象状态 $\hat s$，如果它的**出度已经接近完整**（观察过大部分可能动作），那么"未观察到"的 $(\hat s, a)$ 对就被视为"动作 $a$ 在此处真的不可执行"的强证据。

形式化：$\hat N \subseteq \hat S \times A$ 是满足某种局部覆盖准则的 $(\hat s, a)$ 集合。

## 作用

为算子学习中的 **precondition 提供负样本**——CP-SAT 求解时会强制 $a$ 在 $\hat s$ 上的某个谓词违反前置条件（公式 11），否则付出松弛代价 $\eta$。

## 与闭世界假设的关系

是 [[Closed-World Assumption]] 的**局部**版本：只在出度饱和的状态上做 CWA，避免对未充分探索的状态做错误的负证据假设。

## 与本文关系

是 [[STRIPS-WM]] 能学到正确 precondition 的关键机制。Theorem 10 的 sound & complete 保证就建立在 $\hat N$ 的选择上。

## 相关概念

- [[STRIPS]]
- [[Closed-World Assumption]]
- [[CP-SAT]]
- [[STRIPS-WM]]
