---
type: concept
aliases: [目标条件结构化灵巧策略, Target-Conditioned Policy, Structured Dexterous Policy]
---

# Target-Conditioned Structured Dexterous Policy

## 定义

[[DexFuture]] 低层策略 $\pi_\phi$:把双手灵巧手 + tool + object 表示为**per-link + scene 结构化 token**,输入插值后的未来 target $\tilde{g}_{t+\delta}$,输出动作高斯分布,通过 PPO 训练。架构灵感来自 [[PhysGraph]],但不再依赖 demo GT target。

## 数学定义

$$
a_t \sim \pi_\phi(\cdot|s_t,\ \tilde{g}_t) = \mathcal{N}\!\left(\mu_\phi(s_t, \tilde{g}_t),\ \Sigma_\phi\right)
$$

其中:
- $s_t$: 当前 robot-object 结构化状态
- $\tilde{g}_t$: 高层预测 target 的线性插值结果
- $\mu_\phi$: transformer encoder + policy head 输出的均值
- $\Sigma_\phi$: 可学习对角协方差

## 训练 reward（PPO）

$$
r_t = \sum_{m \in \mathcal{M}} \beta_m \exp\!\left(-\alpha_m\, d_m(s_t,\ s^{demo}_t)\right) - \beta_e \|a_t\|^2
$$

- $\mathcal{M}$: 多模态监督集合（wrist / fingertip / object pose 等）
- $\beta_e$: 动作 L2 正则

## 结构化 token 组成

- $\{z^{hnd}_i\}_{i=1}^{N_h}$: 每个 hand link 的 token（per-link 表示）
- $z^{tool}, z^{obj}$: tool 与 object 的 anchor-aligned scene token
- 与 target $\tilde{g}_t$ 拼接,送 transformer encoder

## 与 [[PhysGraph]] 的关系

|  | [[PhysGraph]] | DexFuture 低层 |
|--|--------------|--------------|
| 架构 | per-link transformer | per-link transformer |
| target 来源 | demo GT $g^{demo}_{t+h}$ | 高层预测 $\tilde{g}_{t+\delta}$ |
| 部署可用性 | 需要 oracle | **可独立部署** |

## 控制频率

60 Hz,与 [[Receding-Horizon Execution|receding-horizon]] 中的高层频率（~1-5 Hz）解耦。

## 代表工作

- [[DexFuture]]: 提出与训练
- [[PhysGraph]]: 架构灵感

## 相关概念

- [[PhysGraph]]
- [[Future-State Visuomotor Target Predictor]]
- [[Receding-Horizon Execution]]
- [[Action Chunking]]
