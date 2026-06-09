---
type: concept
aliases: [DriveDreamer]
---

# DriveDreamer

## 定义

面向自动驾驶的生成式 [[World Model|世界模型]]：以 BEV layout、文本、动作为条件，扩散生成驾驶视频，目标是把驾驶仿真从重建迁移到生成。

## 核心要点

1. BEV / 文本 / 动作多模态条件
2. 基于扩散，质量较 GAN 时代显著提升
3. 多数版本仍是离线生成，闭环支持有限
4. 与 [[GAIA-1]]、[[Vista]]、[[OmniDreams]] 同处生成式驾驶世界模型一脉

## 代表工作

- DriveDreamer 系列
- 被 [[OmniDreams]] 列为对比 baseline

## 相关概念

- [[Video Diffusion Model]]
- [[Action-Conditioned Video Diffusion]]
- [[Closed-Loop Simulation]]
