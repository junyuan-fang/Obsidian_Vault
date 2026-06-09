---
type: concept
aliases: [Vision-Language-Action, 视觉语言动作模型]
---

# VLA (Vision-Language-Action Model)

## 定义
统一处理视觉、语言指令和机器人动作的多模态大模型，输入图像 + 文本指令，输出机器人动作或动作 token 序列。

## 核心要点
1. **代表**：RT-2, OpenVLA, $\pi_0$, GR00T；
2. **训练**：互联网视觉语言预训练 + 机器人数据 SFT；
3. **挑战**：动作 tokenization、跨实体泛化、闭环延迟。

## 代表工作
- 大量 robot manipulation 工作
- [[GRAIL]] 的 egocentric 视觉策略可视为 VLA 的特化版

## 相关概念
- [[Egocentric Visual Policy]]
- [[Diffusion Policy]]
