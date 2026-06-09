---
type: concept
aliases: [Asymmetric DMD, 非对称 DMD]
---

# Asymmetric DMD

## 定义
DMD 的一个变体：教师与学生的生成范式不同（如教师双向多步，学生因果少步），通过 score 差异对齐分布。

## 数学形式
$$
$\min_\theta D_{KL}(p_\theta^{AR,few}\,\|\,p_{teacher}^{Bi,multi})$
$$

## 核心要点
1. 教师可以是双向 video diffusion（多步）
2. 学生是因果自回归少步生成器
3. 需 self-rollout 让训练分布对齐推理分布

## 代表工作
- [[minWM]]: Stage 3 的核心目标，把双向教师对齐到自回归少步学生

## 相关概念
- [[DMD]]
- [[Causal Forcing]]
