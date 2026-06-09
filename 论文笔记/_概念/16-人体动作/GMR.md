---
type: concept
aliases: [Generalized Motion Retargeting]
---

# GMR

## 定义
人形机器人通用动作重定向方法，把 [[SMPL-X]] 序列映射到目标 humanoid 的关节空间（如 [[Unitree G1]]）。

## 核心要点
1. 解决 [[Morphology Mismatch|形态错配]]；
2. 保留接触约束；
3. 是 [[GRAIL]] / VideoMimic 等流水线的标准组件。

## 代表工作
- Ze et al. 2025
- [[GRAIL]]: SMPL-X → G1 关节空间

## 相关概念
- [[Motion Retargeting]]
- [[Morphology Mismatch]]
