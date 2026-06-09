---
type: concept
aliases: [Vision-Language Model, 视觉语言模型]
---

# VLM

## 定义

把视觉编码器（CLIP / SigLIP 等）与 LLM 通过 projector 对齐，让模型能同时处理图像与文本输入并输出语言/动作的多模态模型。

## 核心要点

1. **典型架构**：Vision Tower + MLP Projector + LLM
2. **训练范式**：图文对预训练 → 指令微调 → RLHF/[[GRPO]]
3. **代表家族**：LLaVA / Qwen-VL / InternVL / Gemini
4. **在具身中的角色**：高层规划器、轨迹/动作生成器

## 代表工作

- [[Qwen3-VL]]: GN-BAE 的 backbone
- LLaVA, InternVL, Gemini
- [[GN0]]: 用 VLM 做导航的高层规划

## 相关概念

- [[Qwen3-VL]]
- [[BEV Memory]]
