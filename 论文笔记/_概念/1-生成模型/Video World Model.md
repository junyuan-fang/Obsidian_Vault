---
type: concept
aliases: [Video World Model, 视频世界模型, Interactive World Model]
---

# Video World Model

## 定义
用视频生成模型作为可交互的环境模拟器：给定动作/相机/控制信号，自回归地生成未来帧，供 agent rollout 或规划。

## 数学形式
$$
$p_\theta(o_{t+1}\mid o_{\le t}, a_{\le t})$
$$

## 核心要点
1. 要求：因果生成、低首帧延迟、可控性
2. 应用：embodied AI 规划、game generation、机器人仿真
3. 与传统 latent world model（如 Dreamer）的区别是直接生成像素/视频 latent

## 代表工作
- [[minWM]]: 首个把双向视频扩散蒸馏为相机可控实时世界模型的全栈方案

## 相关概念
- [[Video Diffusion Model]]
- [[Autoregressive Diffusion]]
