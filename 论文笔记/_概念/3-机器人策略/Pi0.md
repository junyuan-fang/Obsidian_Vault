---
type: concept
aliases: [Pi0, π0, π0 VLA, Pi-0]
---

# π0 (Pi0)

## 定义

Physical Intelligence (PI) 公司发布的 [[VLA]] 基础模型，基于大规模视觉 / 视频预训练的 VLM 加上 flow matching 动作头，输出连续动作 chunk，已成为机器人圈最常用的 VLA 基线之一。后续 π0.5 进一步扩大数据 / 任务覆盖。

## 核心要点

1. 通用机器人基座：覆盖单臂 / 双臂 / 桌面 / 家居等任务。
2. Flow-matching 动作头，输出连续动作 chunk（论文中常用 $H_\pi{=}50$）。
3. 支持冻结作为下游策略 / 评测对象。
4. 在世界模型评测论文里被频繁用作"被评测策略"。

## 代表工作

- [[PiL-World]]：把 π0 作为冻结被评测 VLA，在 3 个双臂任务上测想象 SR vs 真机 SR 的一致性。

## 相关概念

- [[VLA]]
- [[Action Chunking]]
- [[Flow Matching]]
- [[Diffusion Policy]]
