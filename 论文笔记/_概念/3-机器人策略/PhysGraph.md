---
type: concept
aliases: [PhysGraph Policy, Physically-Grounded Graph Transformer]
---

# PhysGraph

## 定义

针对**双手灵巧手 + 工具 + 物体**的 physically-grounded graph-transformer 策略,以 hand link、tool、object 的结构化图作为输入,通过 transformer encoder + policy head 输出双手动作分布。是 [[DexFuture]] 低层策略的直接灵感来源,也是评估中的 oracle baseline。

## 核心特点

1. **per-link tokenization**: 每个 hand link 作为独立 token,显式编码 multi-link 结构
2. **scene token**: tool / object 通过 anchor-aligned 池化得到 scene-level token
3. **target conditioning**: 输入未来 demonstration target $g^{demo}_{t+h}$ 作为强引导信号
4. **policy head**: 输出高斯动作分布 $\mathcal{N}(\mu_\phi, \Sigma_\phi)$,PPO 训练

## 数学形式

$$
a_t \sim \pi^{PhysGraph}_\phi(\cdot | s_t,\ g^{demo}_{t+h})
$$

其中 $g^{demo}_{t+h}$ 来自 demonstration —— 这一项是 PhysGraph 的 "privileged" 假设,真实部署难以获得。

## 局限

- **依赖 GT target**: 一旦去掉 demo target,SR 从 ~66% 跌到 ~7%（[[DexFuture]] Table 1）
- **无未来推断能力**: 不预测未来状态,只跟踪给定目标

## 作为 baseline 在 [[DexFuture]] 中的角色

| 配置 | 平均 SR |
|------|---------|
| PhysGraph (GT target) | **66.52%** (oracle 上限) |
| PhysGraph (No target) | ~7% |
| [[DexFuture]] (Pred target) | **59.69%** (~90% oracle) |

## 代表工作

- 原始 PhysGraph 论文
- [[DexFuture]]: 直接复用 per-link transformer 架构

## 相关概念

- [[Target-Conditioned Structured Dexterous Policy]]
- [[Bimanual Dexterous Manipulation]]
- [[Action Chunking]]
