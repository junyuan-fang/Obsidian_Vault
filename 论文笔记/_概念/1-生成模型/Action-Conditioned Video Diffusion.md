---
type: concept
aliases: [动作条件视频扩散, Action-Conditioned Diffusion]
---

# Action-Conditioned Video Diffusion

## 定义

在 [[Video Diffusion Model|视频扩散模型]] 的条件输入中显式加入动作向量（如转向角、油门、机器人末端速度），使生成的下一帧与施加的动作严格对齐。这是把视频扩散模型变成 [[World-Action Model|世界模型]] 的关键一步。

## 数学形式

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}, o_{<t}, s_t, a_t}\!\left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, o_{<t}, s_t, a_t) \|^2 \right]
$$

## 核心要点

1. 动作 $a_t$ 通过 cross-attention / FiLM / MLP 注入到去噪网络
2. 需要带动作标签的视频数据（如 CAN bus、teleop log）
3. 与 [[Causal Forcing|自回归 + causal forcing]] 联合训练，才能用于 [[Closed-Loop Simulation|闭环 rollout]]

## 代表工作

- [[OmniDreams]]: 21,000 小时带方向盘/油门标签驾驶视频上后训练
- 早期工作: [[GAIA-1]]、[[DriveDreamer]]、[[Vista]]

## 相关概念

- [[Video Diffusion Model]]
- [[World-Action Model]]
- [[Closed-Loop Simulation]]
