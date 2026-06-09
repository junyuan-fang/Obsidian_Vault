---
type: concept
aliases: [WAM, 世界-动作模型]
---

# World-Action Model

## 定义

把"世界模型"（预测下一帧）和"动作模型"（预测下一动作）合并成同一神经网络的设计：内部表征同时承担**动态预测**和**策略输出**，两端共享 backbone，仅通过不同 head 区分。

## 数学形式

$$
\underbrace{p_\theta(o_{t+1} | o_{\le t}, a_{\le t})}_{\text{world head}}\;\;\text{与}\;\;\underbrace{\pi_\theta(a_t | o_{\le t})}_{\text{action head}}\;\text{共享}\;\theta
$$

## 核心要点

1. 世界知识与策略共享一份表征
2. 数据效率高：一个模型既学环境也学策略
3. [[OmniDreams]] 报告:WAM backbone 以 1/5 参数超越 VLA 风格 [[Alpamayo|Alpamayo 1.5]]
4. 是把生成式世界模型推到机器人/驾驶策略层面的统一架构思路

## 代表工作

- [[OmniDreams]]: 用 WAM 既做仿真器也做策略 backbone
- [[AdaWAM]]: 给 WAM 装动态路由器，按 chunk 决定是否触发视觉/文本推理，实现"按需做梦"

## 相关概念

- [[Closed-Loop Simulation]]
- [[Action-Conditioned Video Diffusion]]
- [[Video Diffusion Model]]
