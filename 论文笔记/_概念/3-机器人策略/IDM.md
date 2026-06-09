---
type: concept
aliases: [逆动力学模型, Inverse Dynamics Model]
---

# IDM

## 定义
逆动力学模型（Inverse Dynamics Model），给定当前和未来的观测，推断产生该状态转移所需的动作序列。

## 数学形式

$$
a_t = f_\theta(o_t, o_{t+1})
$$

## 核心要点
1. 与前向动力学模型（预测下一状态）互补
2. 在两阶段世界模型方法中常用：先生成未来视频，再用 IDM 解码动作
3. 缺点是依赖视频生成质量，误差会累积

## 代表工作
- [[Cosmos-Policy]]: 使用 IDM 从生成的视频中解码动作
- [[GigaWorld-Policy]]: 避免使用 IDM，直接从观测预测动作

## 相关概念
- [[WAM]]
- [[Video Prediction]]
