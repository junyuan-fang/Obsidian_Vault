---
type: concept
aliases: [动作空间, Action Representation]
---

# Action Space

## 定义
机器人策略输出动作的数学空间定义，规定了动作的维度、范围和语义（如关节角度、末端执行器位姿增量等）。

## 核心要点
1. 连续动作空间 $a \in \mathbb{R}^n$ 常用于机器人操作
2. 维度取决于机器人自由度：单臂通常 7D（6-DoF + 夹爪），双臂 14D
3. 动作表示方式影响策略学习难度（绝对位姿 vs 增量 vs 关节角度）

## 代表工作
- [[Hi-WM]]: 14 维连续动作空间用于双臂操作

## 相关概念
- [[End-Effector]]
- [[Diffusion Policy]]
