---
type: concept
aliases: [神经重建, NuRec]
---

# Neural Reconstruction

## 定义

用 [[NeRF]] / [[Gaussian Splatting]] 等可微神经表示，从采集的真实驾驶/室外视频中"重建"出可重新渲染的 3D / 4D 场景，作为自动驾驶 [[Closed-Loop Simulation|闭环仿真]] 的底层场景源。NVIDIA Omniverse 中称为 NuRec。

## 核心要点

1. 渲染逼真度高
2. 只能"重现"采集片段，对未见过的天气/动态行为泛化差
3. 与生成式世界模型（如 [[OmniDreams]]）正好互补
4. 通常配合 ego-pose / agent 标注供策略训练

## 代表工作

- NVIDIA Omniverse NuRec
- [[OmniDreams]] 中作为对照——生成式 vs 重建式 [[Closed-Loop Simulation|闭环模拟器]]

## 相关概念

- [[NeRF]]
- [[Gaussian Splatting]]
- [[Closed-Loop Simulation]]
