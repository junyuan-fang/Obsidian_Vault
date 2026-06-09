---
type: concept
aliases: [Qwen3 VL]
---

# Qwen3-VL

## 定义

阿里 Qwen 团队 2025-2026 发布的开源 [[VLM|视觉语言模型]] 系列，Qwen3 LLM + 升级版 vision tower，支持长上下文、高分辨率图像与视频。

## 核心要点

1. **架构**：Qwen3 LLM + 多尺度 vision encoder + 多模态 projector
2. **特性**：长上下文、动态分辨率、强工具调用能力
3. **开源**：HuggingFace 上有多个尺寸（2B/7B/72B 量级）
4. **在具身中**：被 GN-BAE 等用作 backbone，因为预训练于真实照片，与 3DGS 渲染视觉风格接近

## 代表工作

- [[GN0]]: GN-BAE 的 VLM backbone
- Qwen-VL → Qwen2-VL → Qwen3-VL 系列演进

## 相关概念

- [[VLM]]
