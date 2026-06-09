---
type: concept
aliases: [KL Divergence, KL 散度, Kullback-Leibler]
---

# KL Divergence

## 定义
衡量两个概率分布差异的非对称度量，等于交叉熵减熵。

## 数学形式
$$
$D_{KL}(p\|q) = \int p(x) \log\frac{p(x)}{q(x)}\,dx$
$$

## 核心要点
1. 非对称：$D_{KL}(p\|q) \ne D_{KL}(q\|p)$
2. VAE / DMD / RLHF 的核心损失之一
3. 可分解为熵 + 交叉熵

## 代表工作
- [[minWM]]: Asymmetric DMD 的目标即学生与数据分布的 KL

## 相关概念
- [[DMD]]
- [[Score Distillation]]
