---
type: concept
aliases: [流匹配, Conditional Flow Matching, CFM]
---

# Flow Matching

## 定义
一种连续归一化流的训练方法，通过直接回归速度场实现从噪声到数据的确定性映射，相比扩散模型采样效率更高。

## 数学形式

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \Big[ \lVert v_\theta(\mathbf{x}_t, t) - \mathbf{u}_t \rVert^2 \Big]
$$

其中插值路径 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$，目标速度 $\mathbf{u}_t = \boldsymbol{\epsilon} - \mathbf{x}_0$。

## 核心要点
1. 学习速度场而非噪声预测，采样通过 ODE 求解
2. 线性插值路径简单高效，可用 Euler 方法少步采样
3. 训练稳定性优于 Score-based 方法

## 代表工作
- [[MultiWorld]]: 基于 Flow Matching 的视频世界模型
- [[GigaWorld-Policy]]: 使用流匹配同时训练动作和视频预测

## 相关概念
- [[ODE]]
- [[DiT]]
