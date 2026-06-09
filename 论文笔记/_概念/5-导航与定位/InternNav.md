---
type: concept
aliases: [InternNav-S2]
---

# InternNav

## 定义

上海 AI Lab 系列导航基础模型，dual-system 架构（高层 VLM + 低层执行器），其中 S2 是较强的 stage-2 微调版。

## 核心要点

1. dual-system 思想：上层规划 + 下层执行
2. NavDP-style 低层
3. 包含开源 Action Expert 训练 pipeline，被 GN-BAE 沿用

## 代表工作

- InternNav 系列
- [[GN0]]: GN-BAE 沿用 InternNav 的 Action Expert 训练流程，并作为 baseline

## 相关概念

- [[NavDP]]
- [[Vision-Language Navigation]]
