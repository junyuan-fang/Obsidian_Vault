---
type: concept
aliases: [RoboTwin 2.0, RoboTwin2.0]
---

# RoboTwin

## 定义

双手机器人仿真 benchmark，覆盖 50 个 manipulation 任务，支持 *clean* 和 *randomized* 两档评测（后者随机化物体姿态/颜色/光照等），是当前 bimanual VLA 评测的主流平台。

## 核心要点

1. **任务规模**: 50 manipulation 任务，覆盖 pick / place / stack / pour / unscrew 等
2. **两档难度**: Clean 全条件可控，Randomized 加视觉/物理扰动
3. **指标**: 单任务成功率 + 平均成功率
4. **典型 SOTA 量级**: 2026 年顶层方法 clean ≈ 92–93%, randomized ≈ 90–91%

## 代表工作

- [[WLA]]: 2B WLA-0 在 Clean 92.94% / Randomized 90.02% 击穿 SOTA
- [[AdaWAM]]: Hard subset 88.43% / Overall 93.11%（路由式 WAM 新 SOTA）
- π₀.₅ / Motus / Fast-WAM 均以 RoboTwin 为主对比表

## 相关概念

- [[LIBERO]]
- [[VLA]]
- [[Embodied AI]]
