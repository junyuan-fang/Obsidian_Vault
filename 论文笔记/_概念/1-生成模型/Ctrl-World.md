---
type: concept
aliases: [Ctrl-World, Controllable World Model for VLA]
---

# Ctrl-World

## 定义

Ctrl-World 是 PiL-World 之前的 SOTA policy-compatible 世界模型，把 VLA 预测的动作作为条件去生成未来观测，用于 VLA 策略评估和数据增广。它支持闭环 rollout，但相对 PiL-World 存在跨轮上下文丢失、训练只见成功轨迹等问题，导致 imagined success rate 系统性偏低。

## 核心要点

1. 闭环可用：可以把生成观测反馈给策略再次查询。
2. 训练数据偏成功示范，缺乏失败轨迹覆盖。
3. 跨轮没有显式的 latent history memory，长 rollout 容易漂移。
4. 在 PiL-World 论文中作为主要基线，三任务平均 $|\Delta SR|{=}63.2\%$ 而 PiL-World 仅 12.0%。

## 代表工作

- [[PiL-World]]：把 Ctrl-World 作为对照，量化"接口对齐 + history memory + 失败轨迹"的增益。

## 相关概念

- [[World Model]]
- [[Video World Model]]
- [[Action-Conditioned Video Diffusion]]
- [[Closed-Loop Simulation]]
- [[Policy-in-the-Loop]]
