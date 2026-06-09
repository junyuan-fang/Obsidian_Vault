---
type: concept
aliases: [双手灵巧操作, Bimanual Dexterous, 双手灵巧手, Bimanual Tool Use]
---

# Bimanual Dexterous Manipulation

## 定义

使用**两只多自由度灵巧手**（每只 16-22 DoF）协同完成接触受限、高动态、长时程的工具使用 / 物体操控任务（切、剪、倒、擦、搅等）。区别于单臂 + 平行夹爪范式，bimanual dexterous 需要同时处理 hand-tool-object 三者的运动学耦合与高频接触。

## 核心挑战

1. **高维动作空间**: 单手 22 DoF × 2 + wrist 6DoF × 2 = ~56 DoF,对学习与规划都是高维诅咒
2. **接触富集**: 工具与对象之间持续接触,误差累积易导致工具脱手/失稳
3. **双手协同**: 两手必须同时满足任务条件（如剪刀闭合 + 纸张固定）
4. **长时程**: 切、倒等任务往往 5-10 秒,需要长 horizon 一致性
5. **缺乏 demo 数据**: 高质量双手 demo 采集昂贵（动捕/teleoperation）

## 主流路线

| 路线 | 代表 | 特点 |
|------|------|------|
| Demo-driven target tracking | [[PhysGraph]] / [[ManipTrans]] | 用 demo 未来 target 作 oracle 条件 |
| Hierarchical target prediction | [[DexFuture]] | 从 visuomotor 历史预测 future target |
| Action-conditioned world model + planning | [[DexWM]] | 在 latent 空间用 [[Cross-Entropy Method (CEM)|CEM]] 规划 |
| 直接 [[Action Chunking|action chunking]] | ACT-bimanual | 端到端预测 chunk |

## 代表数据集

- [[OakInk2]]: 双手 hand-tool-object 接触轨迹
- TACO: 双手工具操作
- ARCTIC: 双手与刚性物体交互

## 代表工作

- [[DexFuture]]: hierarchical future-state target prediction
- [[PhysGraph]]: physically-grounded graph transformer policy
- [[ManipTrans]]: target-conditioned manipulation transformer

## 相关概念

- [[Target-Conditioned Structured Dexterous Policy]]
- [[OakInk2]]
- [[Action Chunking]]
- [[Human-Object Interaction]]
