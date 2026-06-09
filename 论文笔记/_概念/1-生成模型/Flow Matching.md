---
type: concept
aliases: [流匹配, Rectified Flow, Conditional Flow Matching]
---

# Flow Matching

## 定义

一类生成模型训练范式：直接回归从数据分布到噪声分布的"速度场"$v_\theta(x_t, t)$，避免 [[Diffusion Model|扩散模型]] 的多步噪声调度，得到 rectified / 直线路径,推理步数更少。

## 数学形式

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\| v_\theta(x_t, t) - (\epsilon - x_0) \|_2^2\right], \quad x_t = (1-t)x_0 + t\epsilon
$$

预测速度 $v^* = \epsilon - x_0$，即从数据指向噪声的方向。

## 核心要点

1. **直线路径**: 训练后采样可用少量 ODE 步（甚至 1 步）完成
2. **与扩散等价但更简洁**: 训练目标无 SNR 调度，超参更少
3. **Rectified Flow**: 反复 "reflow" 让路径更直，进一步减步数
4. SD3、Flux、[[Cosmos3]] 等大型生成模型已普遍采用

## 代表工作

- Rectified Flow (Liu et al., 2022)：直线化路径
- Flow Matching (Lipman et al., 2023)：通用条件框架
- SD3 / Flux / [[Cosmos3]]：工业级 FM 主干

## 相关概念

- [[Diffusion Model]]
- [[Score Function]]
- [[DMD]]
