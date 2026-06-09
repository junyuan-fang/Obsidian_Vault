---
type: concept
aliases: [未来状态视觉运动目标预测器, Future-State Target Predictor, Visuomotor Target Predictor]
---

# Future-State Visuomotor Target Predictor

## 定义

[[DexFuture]] 提出的**高层目标预测模块** $F_\theta$:从 egocentric RGB + 本体/几何历史预测多步未来 hand-tool-object **target**（而非 action）。是 hierarchical visuomotor policy 中替代"demo 特权 target"的核心组件。

## 数学定义

$$
\hat{g}_{t+h} = F_\theta(\mathcal{O}_{t-K:t};\ \mathcal{H}),\quad h \in \mathcal{H}
$$

其中:
- $\mathcal{O}_{t-K:t} = \{I_\tau, p_\tau\}$: 视觉 + 本体历史
- $\mathcal{H} = \{0, 2, 4, \ldots, 16\}$: 稀疏 horizon 集合
- $\hat{g}_{t+h}$: 第 $h$ 步未来的结构化 target（hand pose / object pose / 关键点）

## 内部组成

1. **结构化 visuomotor tokenization**: hand-link cross-attn + [[FiLM Modulation|FiLM]] + tool/object anchor pooling
2. **[[Horizon-Conditioned Transformer|Horizon-conditioned transformer]]**: [[AdaLN|AdaLN]] 注入 $h$,共享 backbone 输出多 horizon target
3. **Target decoder** $D_\theta$: 把嵌入解码为 policy 接口兼容的 target

## 与传统 action chunking 的区别

| 范式 | 预测对象 | 维度 | 长 horizon 难度 |
|------|---------|------|---------------|
| [[Action Chunking]] | 未来 action 序列 | 高（56+ DoF × H） | 高（误差累积） |
| Future-state target | 未来 state target | 中（结构化） | **中（可学）** |

target 比 action 维度低、更稳定（state 比 control 更平滑）,因此长 horizon 上更容易学。

## 在 hierarchy 中的角色

- 慢频率刷新（每 $J$ 步触发一次）
- 输出稀疏 target,供低层 [[Target-Conditioned Structured Dexterous Policy|policy]] 插值跟踪
- 实现"慢预测 + 快控制"频率解耦

## 代表工作

- [[DexFuture]]: 提出此模块

## 相关概念

- [[Horizon-Conditioned Transformer]]
- [[Target-Conditioned Structured Dexterous Policy]]
- [[Receding-Horizon Execution]]
- [[Action Chunking]]
