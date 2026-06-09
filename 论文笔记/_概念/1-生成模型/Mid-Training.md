---
type: concept
aliases: [中期训练, mid-training]
---

# Mid-Training

## 定义

介于"预训练 (pre-training)"和"后训练 (post-training)"之间的训练阶段，用领域大规模无监督/弱监督数据继续训练通用基础模型，让它从"通用世界"过渡到"目标领域"，再进入对齐/指令/任务后训练。

## 核心要点

1. 数据：领域大规模 raw 数据（如驾驶视频、机器人 teleop）
2. 目标：注入领域视觉/动作分布，不丢失通用先验
3. 通常配合 LoRA / 全参数低 lr 进行
4. 是把通用 [[Cosmos]] / 通用 LLM 落地领域应用的关键阶段

## 代表工作

- [[OmniDreams]]: Cosmos → 21,000h 驾驶视频 mid-training → 动作条件 post-training

## 相关概念

- [[Cosmos]]
- [[Video Diffusion Model]]
