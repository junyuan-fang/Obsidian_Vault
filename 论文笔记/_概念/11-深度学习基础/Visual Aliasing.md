---
type: concept
aliases: [Perceptual Aliasing, 视觉混叠]
---

# Visual Aliasing

## 定义

视觉表征中的一种失败模式：**两张语义不同的图像被映射到相同的 latent 表示**（embedding 或离散 code）。又称 perceptual aliasing。

## 在视觉规划中的危害

- 不同 STRIPS 状态被聚成同一 [[FSQ]] code → 同一 code 下同一动作产生不同 successor → latent nondeterminism
- 导致学到的 precondition / effect 不可靠
- BFS / A\* 规划时返回错误路径

## 缓解策略

| 策略 | 思路 |
|---|---|
| Predictive Separation Loss | 检测到 nondeterminism 时拉开 pre-quantization latent |
| 增加 FSQ 维度 / level | 提高离散容量 |
| 增加 [[Inverse Dynamics]] 辅助 | 强迫保留动作可分性 |
| Multi-view 输入 | 减少单视角混淆（[[STRIPS-WM]] 未采用） |

## 与本文关系

[[STRIPS-WM]] 论文 Section 7 把 visual aliasing 列为主要 limitation：当视觉编码器把不同任务状态混叠时，Stage 2 解不出有效算子。

## 相关概念

- [[FSQ]]
- [[Predictive Separation Loss]]
- [[Image-Grounded Task Graph]]
