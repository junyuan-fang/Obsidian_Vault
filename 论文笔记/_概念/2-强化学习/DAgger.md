---
type: concept
aliases: [Dataset Aggregation]
---

# DAgger

## 定义

Ross et al., 2011 提出的在线模仿学习算法：让当前策略 rollout 收集状态，专家对 rollout 状态进行 relabel，逐步把训练集"打开"到策略真实访问到的状态分布上，从而缓解 [[Covariate Shift|协变量偏移]]。

## 数学形式

$$
\mathcal{L}_{DAgger}(\theta) = -\mathbb{E}_{(x,y^*)\sim \mathcal{D}_{DAgger}} \left[ \frac{1}{L}\sum_{t=1}^{L} \log \pi_\theta(y_t^* \mid y_{<t}^*, x) \right]
$$

其中 $\mathcal{D}_{DAgger}$ 由当前策略 rollout + 专家 relabel 得到。

## 核心要点

1. **目标不是降 loss**：而是让策略的支持集对齐真实分布
2. **桥接 SFT 与 RL**：纯 SFT 后熵太低，直接上 RL 探索失败；DAgger 把分布"摊平"提供更好的起点
3. **对 oracle 依赖强**：需要能对任意状态给出动作标签的专家（如 A\*、MPC、规则）

## 代表工作

- [[GN0]]: 收集 284K 闭环样本，做 SFT 与 DAPO 之间的关键桥梁
- Diffusion-DAgger: 操作领域的延伸

## 相关概念

- [[Covariate Shift]]
- [[SFT]]
- [[DAPO]]
- [[GRPO]]
