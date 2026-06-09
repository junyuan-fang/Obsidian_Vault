---
type: concept
aliases: [VLN, 视觉语言导航]
---

# Vision-Language Navigation

## 定义

让智能体根据自然语言指令在 3D 环境中移动并完成任务的研究范式，是具身智能的核心子领域。

## 核心要点

1. **两类设定**：
   - **离散 VLN**：在预定义全景图节点上做选择（R2R / R4R），简化但不真实
   - **连续 VLN**：在 metric 环境中输出低层连续动作（VLN-CE / VLN-PE / GN-Bench）
2. **三类任务**：goal-nav / instruction-following / human-following
3. **核心挑战**：长程一致性、语言-视觉-动作对齐、协变量偏移、Sim-to-Real
4. **现代主流**：VLM 端到端 + dual-system（VLM 规划 + 低层执行器）

## 代表工作

- [[GN0]]: 用 3DGS 把数据/仿真/训练/部署统一成 pipeline
- [[NaVid]] / [[UniNaVid]]: Video-LLM 单目 RGB VLN
- [[ETPNav]]: 全景图 + topological map
- [[StreamVLN]] / [[DualVLN]] / [[CorrectNav]]: SOTA RGB-only VLN-CE
- [[GaussNav]]: 3DGS 用作导航记忆

## 相关概念

- [[VLN-CE]] / [[VLN-PE]]
- [[BEV Memory]]
- [[DAgger]]
