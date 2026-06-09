---
type: concept
aliases: [Receding Horizon, 滚动时域执行, MPC-style Execution]
---

# Receding-Horizon Execution

## 定义

一种**慢预测 + 快跟踪**的分层执行策略:高层每隔 $J$ 步预测一段未来轨迹（长度 $H$）,中间步用插值后的 sub-target 喂给低层高频控制器。本质上与 [[Model Predictive Control|MPC]] 的滚动时域思想一致,但常被用在 learning-based hierarchical policy（如 [[DexFuture]]）。

## 执行流程

设触发时刻序列 $t_0 < t_1 < \ldots$,每个 $t_j$ 刷新一次目标:

$$
\begin{aligned}
\hat{g}_{t_j:t_j+H} &= F_\theta(\mathcal{O}_{t_j-K:t_j};\ \mathcal{H}) \\
a_{t_j+\delta} &\sim \pi_\phi\!\left(\cdot|s_{t_j+\delta},\ \mathrm{Interp}(\hat{g}_{t_j:t_j+H},\ \delta)\right)
\end{aligned}
$$

其中 $0 \le \delta < t_{j+1} - t_j$,低层在中间步用 linear interp 获得密集 target。

## 关键参数

| 参数 | 含义 | 典型值（[[DexFuture]]） |
|------|------|----------------------|
| 高层触发间隔 $J$ | 多久重算一次 target | 数十步 |
| Horizon $H$ | 预测多远 | 16 |
| 低层控制频率 | 跟踪频率 | **60 Hz** |
| 高层频率 | target 刷新 | 约 1-5 Hz |

## 优点

1. **频率解耦**: 慢的语义预测不阻塞快的接触控制
2. **闭环纠错**: 每隔 $J$ 步重新读 obs,长 horizon 也不会累积误差
3. **鲁棒**: 即使中间 target 不完美,下次触发时会自动校正

## 在 [[DexFuture]] 中的角色

避免每步重算高层 transformer 的算力开销,同时保留接触控制需要的 60 Hz 响应。这是把 0.24 Hz 慢规划（[[Cross-Entropy Method (CEM)|CEM]]）路线打败 ~250× 的关键设计。

## 代表工作

- [[Model Predictive Control|MPC]]: 经典控制中的滚动时域思想
- [[DexFuture]]: hierarchical policy 中的 receding-horizon 执行

## 相关概念

- [[Model Predictive Control]]
- [[Future-State Visuomotor Target Predictor]]
- [[Target-Conditioned Structured Dexterous Policy]]
