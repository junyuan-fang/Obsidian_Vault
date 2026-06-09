---
type: concept
aliases: [域随机化, DR]
---

# Domain Randomization

## 定义
在仿真训练中对视觉、动力学、传感器参数等做大幅度随机扰动，使策略在仿真分布中学到对真实环境分布鲁棒的行为，是 Sim-to-Real 的核心技巧。

## 核心要点
1. **视觉 DR**：纹理 / 光照 / 相机噪声；
2. **动力学 DR**：质量 / 摩擦 / 关节阻尼；
3. **观测 DR**：观测延迟 / 噪声；
4. 通常配合 system identification 同时使用。

## 代表工作
- OpenAI Dactyl (2018)
- [[GRAIL]]: 训练 egocentric 视觉策略时用 DR 做 sim2real

## 相关概念
- [[Sim2Real Gap]]
- [[Egocentric Visual Policy]]
- [[PPO]]
