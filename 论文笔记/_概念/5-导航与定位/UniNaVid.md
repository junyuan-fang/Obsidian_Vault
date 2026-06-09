---
type: concept
aliases: [Uni-NaVid]
---

# UniNaVid

## 定义

Zhang et al., 2025 提出的端到端 video-based 导航模型，统一短期与长期导航历史的隐式记忆形式。

## 核心要点

1. 用 video encoder 编码短期 + 长期历史
2. 单一架构覆盖多种 VLN 任务
3. SFT 路线，未做闭环 RL

## 代表工作

- Zhang et al., 2025
- [[GN0]]: 作为 GN-Bench / VLN-CE 上的对比基线（包含 SFT-on-GN-Matrix 变体）

## 相关概念

- [[Vision-Language Navigation]]
- [[NaVid]]
