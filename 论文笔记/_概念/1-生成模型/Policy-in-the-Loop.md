---
type: concept
aliases: [Policy-in-the-Loop, PiL, 策略闭环, PiL Evaluation]
---

# Policy-in-the-Loop (PiL)

## 定义

一种闭环评测 / 仿真范式：在 imagined rollout 中，世界模型不断生成未来观测，**把观测反馈给被评测策略，让策略重新查询动作**，再生成下一段观测，循环往复。与之相对的是 open-loop，沿一段预定动作走，不重新查策略。PiL 是真机部署的最真实近似，因此 imagined SR 才有意义。

## 核心要点

1. 接口对齐：observation → policy → action chunk → observation。
2. 强调"分布偏移可补"——策略偏离后世界模型能跟进，open-loop 评测无法捕捉。
3. 对世界模型要求更高：要支持任意中间观测作为起点继续生成。
4. 通常按 chunk 粒度循环（一次出 $H_\pi$ 步动作）。

## 代表工作

- [[PiL-World]]：第一篇明确把"PiL 接口对齐"作为世界模型设计目标的工作。
- [[Ctrl-World]] / WorldEval / Hi-WM 等部分支持 PiL，但未对齐 chunk 粒度或缺失上下文。

## 相关概念

- [[Closed-Loop Simulation]]
- [[World Model]]
- [[VLA]]
- [[Action Chunking]]
