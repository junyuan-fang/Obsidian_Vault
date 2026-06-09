---
type: concept
aliases: [协变量偏移, Distribution Shift in IL]
---

# Covariate Shift

## 定义

模仿学习中，训练数据由专家策略采样，但部署时策略遵循自身策略；两者的状态分布不一致，导致小误差被时序放大、长程崩溃。

## 数学描述

设训练分布 $\rho_{expert}(s)$ 与部署分布 $\rho_\pi(s)$，[[SFT]] 只保证 $\mathbb{E}_{s \sim \rho_{expert}}[\ell(\pi(s), \pi^*(s))]$ 小，但部署损失 $\mathbb{E}_{s \sim \rho_\pi}[\ell]$ 在长 horizon 下指数级放大。

## 核心要点

1. **后果**：闭环执行偏离越偏越远
2. **缓解手段**：[[DAgger]]、在线 RL、噪声注入、recovery 标签
3. **VLN 中表现**：进入未访问的过道或卡死在墙边

## 代表工作

- DAgger: Ross et al., 2011 经典解
- [[GN0]]: SFT → DAgger 阶段专门用 284K 闭环 relabel 缓解此问题

## 相关概念

- [[DAgger]]
- [[SFT]]
