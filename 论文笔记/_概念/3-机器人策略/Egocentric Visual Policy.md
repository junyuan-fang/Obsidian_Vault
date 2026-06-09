---
type: concept
aliases: [第一人称视觉策略, 头戴相机策略]
---

# Egocentric Visual Policy

## 定义
以机器人头戴 / 胸前相机第一人称视角的 RGB（有时含深度）为主要输入的控制策略，输出关节动作或 latent token，常用于 sim-to-real 部署。

## 核心要点
1. **输入**：第一人称 RGB + 本体感知；
2. **输出**：动作 / latent / waypoint；
3. **训练**：常通过 teacher-student / DAgger 从 privileged policy 蒸馏；
4. **关键挑战**：[[Domain Randomization|域随机]] 以缩小 [[Sim2Real Gap]]。

## 代表工作
- [[GRAIL]]: 蒸馏 latent 策略为 RGB 输入的视觉策略
- VideoMimic, HumanX

## 相关概念
- [[Sim2Real Gap]]
- [[Domain Randomization]]
- [[Unitree G1]]
