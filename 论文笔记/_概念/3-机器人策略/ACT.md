---
type: concept
aliases: [Action Chunking with Transformers]
---

# ACT

## 定义
基于 Transformer 的动作分块策略，一次预测一组连续动作（chunk），减少累积误差。

## 核心要点
1. 使用 CVAE + Transformer 架构
2. Action chunking 减少决策频率，提高平滑性
3. 常用于桌面操作任务

## 代表工作
- Zhao et al. 2023: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

## 相关概念
- [[Diffusion Policy]]
- [[VLA]]