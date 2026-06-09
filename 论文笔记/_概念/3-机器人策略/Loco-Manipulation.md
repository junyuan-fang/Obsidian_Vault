---
type: concept
aliases: [移动操作, 运动操控, Locomotion + Manipulation]
---

# Loco-Manipulation

## 定义
移动操作（Loco-Manipulation）指机器人在移动（行走 / 攀爬 / 跨地形）的同时完成物体操控（抓取 / 放置 / 推拉），是 humanoid / quadruped 机器人最具挑战性的全身控制任务。

## 核心要点
1. 需要同时协调下肢（locomotion）和上肢（manipulation）控制；
2. 全身控制器需保持平衡的同时输出操控动作；
3. 数据稀缺：teleoperation / mocap 都难以大规模收集；
4. 评估涉及移动成功率、抓取成功率、双任务联合成功率。

## 代表工作
- [[GRAIL]]: 用 VFM + 3D 资产合成 20K+ loco-manipulation 序列
- [[HDMI]]: humanoid manipulation 基线
- [[ResMimic]]: 残差模仿基线

## 相关概念
- [[Human-Object Interaction]]
- [[Whole-Body Controller]]
- [[Unitree G1]]
