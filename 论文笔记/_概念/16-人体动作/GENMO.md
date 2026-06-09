---
type: concept
aliases: [Generalist Motion Estimation]
---

# GENMO

## 定义
RGB 单视频人体动作估计模型，从视频中预测每帧 [[SMPL-X]] 身体参数与全局轨迹。

## 核心要点
1. 输入：单视角 RGB；
2. 输出：SMPL-X 序列 + 骨盆世界系轨迹；
3. 通常配合 [[WiLoR]] 补充手部参数。

## 代表工作
- Li et al. 2025
- [[GRAIL]]: 用 GENMO 提供身体初始化

## 相关概念
- [[SMPL-X]]
- [[WiLoR]]
- [[MoGe-2]]
