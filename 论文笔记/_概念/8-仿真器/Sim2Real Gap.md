---
type: concept
aliases: [Sim-to-Real Gap, 仿真到现实差距]
---

# Sim2Real Gap

## 定义

仿真环境与真实世界之间在视觉、动力学、传感器噪声、行为分布上的差异，导致仿真训练的策略迁移到真实部署时性能下降。

## 核心要点

1. 视觉差距：纹理、光照、动态范围
2. 动力学差距：摩擦、延迟、噪声
3. 行为差距：其他 agent 的真实反应分布
4. 生成式世界模型（如 [[OmniDreams]]）和 [[Neural Reconstruction|NuRec]] 的核心动机之一就是缩小视觉差距

## 代表工作

- 几乎所有具身智能/自动驾驶策略论文都讨论该问题
- [[OmniDreams]] 用大规模真实驾驶视频后训练逼近真实分布

## 相关概念

- [[Closed-Loop Simulation]]
- [[CARLA]]
- [[Neural Reconstruction]]
