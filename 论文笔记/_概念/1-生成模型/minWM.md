---
type: concept
aliases: [Minimal World Model]
---

# minWM

## 定义

Vault 中已有对应论文笔记 [[minWM]]（论文笔记/1-世界模型与视频生成/minWM.md），讨论**最小化、可证明的世界模型**的设计原则。

## 与本文关系

[[STRIPS-WM]] 与 minWM 同属"minimal & provable world model"的研究方向，但走完全不同的路线：

| 维度 | minWM | [[STRIPS-WM]] |
|---|---|---|
| 表征空间 | latent / continuous | 0/1 谓词向量 |
| 规划方法 | latent dynamics rollout | 经典 STRIPS 规划器 |
| 形式保证 | 抽象 minimality | sound & complete @ zero-slack |

## 相关概念

- [[World Model]]
- [[Video World Model]]
- [[Dreamer]]
