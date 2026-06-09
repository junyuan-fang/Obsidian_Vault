---
type: concept
aliases: [DMD, Distribution Matching Distillation, 分布匹配蒸馏]
---

# DMD

## 定义
把多步扩散教师蒸馏成单步/少步生成器，目标是让学生输出分布与真实数据分布的 KL 散度最小，用两个 score 网络的差代替显式梯度。

## 数学形式
$$
$\nabla_\theta D_{KL}(p_\theta \| p_{data}) = -\mathbb{E}[(s_{real} - s_{fake})\partial x/\partial \theta]$
$$

## 核心要点
1. 真 score：来自冻结的扩散教师
2. 假 score：来自一个可训练的 critic，需与学生同步更新
3. 代表工作：DMD/DMD2

## 代表工作
- [[minWM]]: Stage 3 用 Asymmetric DMD 把双向教师蒸到自回归学生

## 相关概念
- [[Asymmetric DMD]]
- [[Score Distillation]]
- [[Score Function]]
