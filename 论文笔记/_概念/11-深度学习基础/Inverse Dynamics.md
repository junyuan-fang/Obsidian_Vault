---
type: concept
aliases: [Inverse Dynamics Model, Inverse Model, IDM]
---

# Inverse Dynamics

## 定义

给定相邻两帧 $(s_t, s_{t+1})$ 或 $(o_t, o_{t+1})$，预测中间执行的动作 $a_t$ 的模型：

$$
\hat a_t = h_\psi(z_t, z_{t+1})
$$

与正向动力学（forward dynamics）$f: (z_t, a_t) \to z_{t+1}$ 互为对偶。

## 在自监督表征学习中的角色

作为**辅助任务**强迫学到的 latent 表征保留动作可分性：

- 防止表征塌缩到 trivial 常量（forward dynamics + inverse dynamics 联合训练）
- 提取动作相关的特征，过滤动作无关的视觉噪声

## 与本文关系

[[STRIPS-WM]] Stage 1 在 [[JEPA]] 主损失之外加了 inverse dynamics 辅助损失 $\mathcal{L}_{\text{inv}} = \text{CE}(\hat a_t, a_t)$，防止 [[FSQ]] code 把不同动作的转移混淆。

## 代表工作

- AgentNet (Pathak et al. 2017): 用 inverse dynamics 学好奇心驱动
- Video Pretraining (VPT, OpenAI 2022): 用 inverse dynamics 把无标签视频转为动作监督
- [[JEPA]]: 自监督预训练
- [[STRIPS-WM]]: 视觉规划

## 相关概念

- [[JEPA]]
- [[Predictive Separation Loss]]
