---
type: concept
aliases: [VLN Physics-Enabled]
---

# VLN-PE

## 定义

VLN-CE 的物理增强版，引入 Unitree H1 等真实机器人动力学与底盘控制误差，更贴近 sim-to-real（Wang et al., 2025a）。

## 核心要点

1. 在 Habitat 之上加物理仿真层
2. 暴露底盘漂移、转弯滞后、地面摩擦等真机会出现的低层问题
3. 用作 GN-BAE 等模型的物理鲁棒性测试场

## 代表工作

- [[GN0]]: 在 VLN-PE 上验证物理鲁棒性

## 相关概念

- [[VLN-CE]]
- [[Vision-Language Navigation]]
