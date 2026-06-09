---
type: concept
aliases: [IsaacLab, Isaac Sim Lab]
---

# Isaac Lab

## 定义
NVIDIA 推出的基于 Isaac Sim / Omniverse 的机器人学习仿真框架，提供大规模并行环境、GPU 加速物理、域随机化、PPO 等 RL pipeline。

## 核心要点
1. **GPU 并行**：单卡 1k+ 环境；
2. **集成**：Omniverse 物理 + USD + 多机器人 URDF；
3. **替代品**：Isaac Gym（已弃用）、IsaacLab 为继任者。

## 代表工作
- [[GRAIL]]: PPO 64×L40，每卡 1024 envs，30k iter
- 大量 humanoid / locomotion 工作

## 相关概念
- [[PPO]]
- [[Whole-Body Controller]]
- [[Domain Randomization]]
