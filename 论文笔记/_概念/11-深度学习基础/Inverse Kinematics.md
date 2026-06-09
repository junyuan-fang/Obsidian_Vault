---
type: concept
aliases: [IK, 逆运动学]
---

# Inverse Kinematics (IK)

## 定义
给定末端执行器（如手腕 / 末端工具）的目标位姿，反解出关节角度的过程，是机器人控制 / 动作重定向的基础工具。

## 核心要点
1. **解析 IK**：6 DoF 机械臂常见；
2. **数值 IK**：雅可比迭代 / 优化式；
3. **多解**：通常加正则项选择自然姿态；
4. **应用**：[[Motion Retargeting]]、Wrist IK 把 [[MANO]] 接入 [[SMPL-X]]。

## 代表工作
- 经典机器人学基础
- [[GRAIL]]: wrist IK 整合手部参数

## 相关概念
- [[Motion Retargeting]]
- [[MANO]]
- [[SMPL-X]]
