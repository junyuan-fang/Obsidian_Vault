---
type: concept
aliases: [Alpamayo 1, Alpamayo 1.5, Alpamayo 2 Super]
---

# Alpamayo

## 定义

NVIDIA 自动驾驶策略模型系列：

- **Alpamayo 1**: 基础驾驶策略
- **Alpamayo 1.5**: 视觉语言驱动 (VLA) 策略，作为常见对比 baseline
- **Alpamayo 2 Super**: 32B 参数开源推理模型，面向 L4 robotaxi

它们与 [[OmniDreams]] / AlpaGym 共用 [[Cosmos]] 作为世界模型底座。

## 核心要点

1. 一系列尺度递增的 NVIDIA 内部驾驶策略
2. 提供闭环训练/评估的"被评对象"
3. 与 [[OmniDreams]] 的 [[World-Action Model|WAM]] 形成对照——WAM 用 1/5 参数反超 VLA 风格的 Alpamayo 1.5

## 代表工作

- Alpamayo 1 / 1.5 / 2 Super (NVIDIA)
- [[OmniDreams]] 论文将其作为对比基线

## 相关概念

- [[Cosmos]]
- [[World-Action Model]]
- [[Closed-Loop Simulation]]
