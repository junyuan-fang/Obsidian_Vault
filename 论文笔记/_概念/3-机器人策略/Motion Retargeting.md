---
type: concept
aliases: [动作重定向, retargeting]
---

# Motion Retargeting

## 定义
把一个角色 / 人体的动作序列转换到不同骨架（如人形机器人 [[Unitree G1]]）的关节空间上的过程，需处理骨长差异、关节数量差异和接触约束保留。

## 核心要点
1. **关键约束**：脚-地接触、手-物接触尽量保留；
2. **形态错配**（[[Morphology Mismatch]]）是核心难点；
3. **方法**：解析 IK / 优化式（如 GMR）/ 学习式；
4. **质量度量**：MPJPE、接触一致性、可执行性。

## 代表工作
- [[GMR]]: SMPL-X → humanoid 关节
- [[GRAIL]]: 用机器人形态预匹配的模板从源头规避大量错配

## 相关概念
- [[SMPL-X]]
- [[Morphology Mismatch]]
- [[Inverse Kinematics]]
