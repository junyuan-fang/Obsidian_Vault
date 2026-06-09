---
type: concept
aliases: [π₀, pi-zero, Pi Zero]
---

# Pi0

## 定义
Physical Intelligence 提出的通用机器人基础模型，基于预训练视觉-语言模型进行机器人动作预测，支持跨任务和跨机器人迁移。

## 核心要点
1. 基于大规模预训练的 VLM 骨干进行机器人策略学习
2. 支持语言条件化的多任务操作
3. 通过 flow matching 生成连续动作序列

## 代表工作
- Physical Intelligence, "π₀: A Vision-Language-Action Flow Model for General Robot Control" (2024)
- [[Hi-WM]]: 作为策略骨干之一，后训练后在 Route Rope 任务达到 100% 成功率

## 相关概念
- [[VLA]]
- [[Diffusion Policy]]
