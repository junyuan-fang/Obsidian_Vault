---
type: concept
aliases: [常微分方程, Ordinary Differential Equation, Neural ODE]
---

# ODE

## 定义
常微分方程，在生成模型中用于描述从噪声到数据的连续变换路径，通过数值求解器（如 Euler、RK4）进行采样。

## 数学形式

$$
\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}_t, t), \quad \mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})
$$

## 核心要点
1. Flow Matching 和 Continuous Normalizing Flow 的采样过程本质是 ODE 求解
2. 确定性采样（vs SDE 的随机采样），支持精确似然计算
3. 步数-质量 trade-off：更多步数更精确但更慢

## 代表工作
- [[MultiWorld]]: 通过 ODE 采样生成视频帧

## 相关概念
- [[Flow Matching]]
