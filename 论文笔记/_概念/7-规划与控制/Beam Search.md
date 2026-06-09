---
type: concept
aliases: [束搜索]
---

# Beam Search

## 定义

启发式搜索算法的一种，每一步只保留得分前 $k$（beam width）的候选继续扩展，其余剪枝。是 BFS / best-first 的内存受限近似。

## 在世界模型规划中的角色

[[Video World Model]] / latent world model 做 rollout 规划时，常用 beam search 在动作序列上展开：每步保留 $k$ 条最高奖励的部分轨迹。

## 与本文关系

[[STRIPS-WM]] 的 baseline **WM-Rollout** 在 [[FSQ]] latent code 上做 beam search 规划；但因 latent 动力学误差累积，长 horizon 失败率高。

## 缺点

- **不完备**: 可能裁掉最优解
- **次优**: 局部最大化
- **依赖启发**: 评分函数错误就会引入系统偏差

## 相关概念

- [[BFS]]
- [[Video World Model]]
- [[A* Search]]
