---
type: concept
aliases: [机器人操作, manipulation]
---

# Robot Manipulation

## 定义

机器人用末端执行器（夹爪、灵巧手）与环境交互，完成抓取、放置、装配、推拉、倒水等物理任务的统称。是 [[Embodied AI|具身智能]] 的核心问题之一。

## 核心要点

1. 输入：相机/触觉/本体状态；输出：末端动作
2. 主流方法：模仿学习、扩散策略、VLA
3. 世界模型类方法（如 [[OmniDreams]] 思路推广到机器人）可作为闭环训练环境
4. 仿真器：[[CARLA]] 对应驾驶，IsaacLab / MuJoCo 对应操作

## 代表工作

- 经典：Diffusion Policy、RT-2、π0
- 与世界模型结合方向（如 [[OmniDreams]] 局限性章节的展望）

## 相关概念

- [[Embodied AI]]
- [[World-Action Model]]
- [[Closed-Loop Simulation]]
