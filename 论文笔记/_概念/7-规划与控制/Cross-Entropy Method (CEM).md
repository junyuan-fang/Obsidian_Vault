---
type: concept
aliases: [CEM, 交叉熵方法, Cross-Entropy Method, CEM Planning]
---

# Cross-Entropy Method (CEM)

## 定义

一种 **sampling-based 数值优化方法**,在 model-based RL / planning 中广泛用于在 world model 上做轨迹优化（Trajectory Optimization）。每轮采样候选动作序列 → 用模型 rollout → 选 elite → refit 分布 → 重复迭代。

## 数学流程

设候选动作序列分布 $\mathcal{N}(\mu, \Sigma)$:

1. **采样**: $\{a^{(n)}_{t:t+H}\}_{n=1}^{N} \sim \mathcal{N}(\mu, \Sigma)$
2. **评估**: $J^{(n)} = \text{cost}(\text{WorldModel}.\text{rollout}(s_t, a^{(n)}))$
3. **选 elite**: 选 cost 最低的 top-$K$
4. **Refit**: $\mu \gets \text{mean}(\{a^{(n^\star)}\})$,$\Sigma \gets \text{cov}(\{a^{(n^\star)}\})$
5. **重复**: 多次 CEM 迭代后,执行第一步动作,replan

## 在机器人中的常见用法

- **[[Model Predictive Control|MPC]] inner loop**: PETS / PlaNet / Dreamer 等用 CEM 优化短 horizon 动作
- **action-conditioned [[World Model|world model]] planning**: 如 [[DexWM]] 在 latent 空间 rollout 候选,与 goal latent 比 MSE

## 优缺点

| 优点 | 缺点 |
|------|------|
| 不需可微 dynamics | **采样数 N 随动作维度爆炸** |
| 易于并行 | **慢**: 多次 rollout + 多次迭代 |
| zero-shot 适配新任务 | 高维长 horizon 失效（[[DexFuture]] 实证 H16 完全崩盘） |

## 与 DexFuture 的对比

[[DexFuture]] 的核心论点之一就是 CEM 在双手灵巧操作（56+ DoF）+ 长 horizon 下不可行:
- [[DexWM]] + CEM (H1, finetune): SR 69.57%, **0.24 Hz**
- [[DexWM]] + CEM (H16): **SR 0%**（动作空间太高维）
- [[DexFuture]]: SR 83.49%, **60 Hz**(~250× 快)

## 代表工作

- PETS (Chua et al.): MPC + CEM
- PlaNet / Dreamer: latent CEM
- [[DexWM]]: action-conditioned WM + CEM
- [[DexFuture]]: 反向论证 CEM 不适合双手灵巧操作

## 相关概念

- [[World Model]]
- [[DexWM]]
- [[Model Predictive Control]]
