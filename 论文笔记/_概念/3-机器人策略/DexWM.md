---
type: concept
aliases: [Dexterous World Model, DexWM CEM]
---

# DexWM

## 定义

**action-conditioned world model** for dexterous manipulation,在 latent 空间根据当前观测 + 候选动作序列 rollout 未来 latent,与 goal-image latent 计算 MSE,通过 [[Cross-Entropy Method (CEM)|CEM]] 优化候选动作。是 [[DexFuture]] 主要对比的 model-based planning baseline。

## 数学流程

1. 采样 $N$ 条候选动作序列 $\{a^{(n)}_{t:t+H}\}_{n=1}^{N}$
2. 用 world model rollout 得到 latent $\{z^{(n)}_{t:t+H}\}$
3. 计算与目标 latent 的 MSE: $J^{(n)} = \|z^{(n)}_{t+H} - z^{goal}\|^2$
4. 选 elite samples $\{a^{(n^\star)}\}$,refit Gaussian
5. 重复 CEM 迭代,执行第一步 action,replan

## 原始配置 vs 适配配置

| 参数 | 原始（8×H100） | 适配（单卡 3090Ti） |
|------|---------------|--------------------|
| Horizon | 3 | 16 |
| CEM 步数 | 10 | 4 |
| Samples/iter | 1024 | 128 |
| Elites | 10 | 16 |
| 频率 | ~低 | **0.24 Hz** |

## 性能数据（[[DexFuture]] Table 4, task 083f7@0）

| 配置 | SR (%) |
|------|-------:|
| Finetune + H1 | 69.57 |
| No-finetune + H1 | 0 |
| Finetune + H16 | 0 |
| No-finetune + H16 | 0 |
| **[[DexFuture]] (Pred, H16)** | **83.49** |

## 核心局限

1. **慢**: 0.24 Hz,无法满足接触控制需求（需 60 Hz）
2. **高维 action 长 horizon 不可解**: H16 + finetune 也崩盘
3. **依赖 latent goal image**: 推理时需要目标观测

## 代表工作

- 原始 DexWM 论文
- [[DexFuture]]: 主要对照对象（Sec 4.5 + Appendix 7.1）

## 相关概念

- [[World Model]]
- [[Cross-Entropy Method (CEM)]]
- [[Bimanual Dexterous Manipulation]]
