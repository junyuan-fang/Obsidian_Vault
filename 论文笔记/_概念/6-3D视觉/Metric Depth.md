---
type: concept
aliases: [度量深度, absolute depth]
---

# Metric Depth

## 定义
单位为米（或其他物理长度单位）的深度估计结果，与"相对深度"（仅保留前后关系）相对。是机器人 / AR 等下游必须的尺度恢复。

## 核心要点
1. **单目难点**：单目缺乏尺度先验；
2. **解法**：背景标定（GRAIL）、传感器 fusion、metric-trained 模型（[[MoGe-2]]）；
3. **应用**：3D 重建、HOI 物理可执行性。

## 代表工作
- [[MoGe-2]]
- [[GRAIL]]

## 相关概念
- [[MoGe-2]]
- [[Chamfer Distance]]
