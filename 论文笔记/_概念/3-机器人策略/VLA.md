---
type: concept
aliases: [Vision-Language-Action Model, 视觉-语言-动作模型]
---

# VLA

## 定义
Vision-Language-Action Model 将视觉观察、语言指令和动作输出统一在一个模型中，实现端到端的机器人策略。

## 核心要点
1. 通常基于预训练 VLM 微调
2. 输入：图像 + 语言指令；输出：机器人动作序列
3. 关键挑战：动作空间的连续性、长程规划

## 代表工作
- [[OpenVLA]]: 开源 VLA
- [[pi-0.5]]: Physical Intelligence 的 VLA
- [[GigaWorld-Policy]]: 以动作为中心的 WAM，兼具 VLA 的高效推理

## 相关概念
- [[VLM]]
- [[WAM]]