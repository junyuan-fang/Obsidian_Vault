---
type: concept
aliases: [DreamerV1, DreamerV2, DreamerV3]
---

# Dreamer

## 定义

Hafner et al. 系列工作（2020–2023），代表 RL 中"learn world model + plan in latent space"的主流路线。

| 版本 | 年份 | 核心进展 |
|---|---|---|
| Dreamer (V1) | 2020 | RSSM (Recurrent State Space Model) + actor-critic in latent |
| DreamerV2 | 2021 | 离散 latent + KL balancing |
| DreamerV3 | 2023 | 同一组超参跨 150+ 任务 SoTA，Atari + Crafter + DMC + Minecraft |

## 架构核心

- **World Model**: RSSM = deterministic GRU $h_t$ + stochastic latent $z_t$
- **Actor / Critic**: 在 latent rollout 上做 PPO-style 训练
- **像素重构**: 用 $z_t$ 重构 $o_t$，提供监督信号

## 与 STRIPS-WM 对比

| 维度 | Dreamer | [[STRIPS-WM]] |
|---|---|---|
| 路线 | RL + pixel rollout | 神经符号 + classical planning |
| 状态 | 连续 latent | 离散 0/1 谓词向量 |
| 规划 | imagination rollout + actor | BFS / A\* 在符号空间 |
| 长 horizon | 误差累积 | 100% |

## 与本文关系

[[STRIPS-WM]] 的 baseline **WM-Rollout** 借鉴 Dreamer-style latent rollout 思路，但用 [[FSQ]] 代替 RSSM；论文证明 latent rollout 长 horizon 退化是结构性缺陷，不只是模型规模问题。

## 相关概念

- [[Video World Model]]
- [[World Model]]
- [[Beam Search]]
