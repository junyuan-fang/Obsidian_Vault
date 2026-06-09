---
type: concept
aliases: [闭环仿真, closed-loop, closed-loop evaluation]
---

# Closed-Loop Simulation

## 定义

策略动作能够**实时影响**后续传感器观测的仿真范式：每个时间步 $\pi(a_t | o_{\le t})$ → 仿真器更新状态 $s_t$ → 仿真器生成下一帧观测 $o_{t+1}$ → 再喂回策略，构成回路。与"开环回放数据"形成对照。

## 数学形式

$$
p(o_{1:T}, a_{1:T} | s_0) = \prod_{t=1}^T p_\theta(o_t | o_{<t}, s_{<t}, a_{<t})\,\pi_\phi(a_t | o_{\le t})\,T(s_t | s_{t-1}, a_{t-1})
$$

## 核心要点

1. 是评估 / 训练自动驾驶或机器人策略的金标准
2. 关键瓶颈：仿真器实时性 + 逼真度
3. 传统方案：CARLA / NuRec；新方案：生成式世界模型如 [[OmniDreams]]

## 代表工作

- [[OmniDreams]]: 基于 [[Cosmos]] 的实时闭环驾驶仿真
- [[CARLA]]: 物理引擎类闭环

## 相关概念

- [[World-Action Model]]
- [[Neural Reconstruction]]
- [[Sim2Real Gap]]
