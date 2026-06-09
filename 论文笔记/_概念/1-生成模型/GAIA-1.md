---
type: concept
aliases: [GAIA-1, Wayve GAIA]
---

# GAIA-1

## 定义

Wayve 提出的早期生成式驾驶 [[World Model|世界模型]]，把驾驶视频生成视为带文本/动作条件的自回归 token 序列建模任务，是"用生成模型当驾驶模拟器"思路的开山之作之一。

## 核心要点

1. 离散 token + 自回归生成
2. 多模态条件：文本、动作、过去帧
3. 开环视频生成为主，未实现严格 [[Closed-Loop Simulation|闭环]]
4. 启发了后续 [[DriveDreamer]]、[[Vista]]、[[OmniDreams]] 等驾驶世界模型

## 代表工作

- GAIA-1 (Wayve, 2023)
- 被 [[OmniDreams]] 视为相关 baseline 之一

## 相关概念

- [[Video Diffusion Model]]
- [[Action-Conditioned Video Diffusion]]
- [[Closed-Loop Simulation]]
