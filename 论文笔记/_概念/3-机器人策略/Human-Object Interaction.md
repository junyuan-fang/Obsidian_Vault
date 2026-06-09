---
type: concept
aliases: [HOI, 人物交互, 人体-物体交互]
---

# Human-Object Interaction (HOI)

## 定义
人-物交互（HOI）指建模 / 重建 / 生成人类与物体之间的接触交互过程，常表示为 4D（人体姿态 + 物体位姿）随时间序列。

## 核心要点
1. **表示**：人体 [[SMPL-X]] 参数 $\Theta^H$ + 物体 6DoF 位姿 $\Theta^O$；
2. **关键约束**：接触一致性（hand-object contact）、无穿透（penetration）、物理可执行；
3. **数据来源**：动捕（GRAB, BEHAVE, ARCTIC）、RGB 重建（PHOSA, CHORE）、VFM 生成（[[DAViD]], [[GRAIL]]）；
4. **下游用途**：机器人模仿学习、动作生成、AR/VR。

## 代表工作
- [[GRAIL]]: 资产先验式 4D HOI 生成
- [[CHOIS]]: 语言+waypoint 条件训练式 HOI
- [[HOIDiff]]: 扩散式 affordance HOI
- [[DAViD]]: VFM-free 图到视频 HOI

## 相关概念
- [[SMPL-X]]
- [[MANO]]
- [[InterMimic]]
