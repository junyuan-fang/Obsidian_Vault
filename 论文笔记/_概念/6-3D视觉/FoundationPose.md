---
type: concept
aliases: [6DoF Object Pose]
---

# FoundationPose

## 定义
通用 6DoF 物体位姿估计与追踪模型，可在已知 3D mesh 的情况下从 RGB(-D) 视频中追踪物体的世界系位姿。

## 核心要点
1. **输入**：3D mesh + RGB 帧（depth 可选）；
2. **输出**：6DoF 位姿；
3. **GRAIL 用法**：合成数据微调 5 epoch，depth 通道置零，从已知首帧前向传播。

## 代表工作
- Wen et al. 2024
- [[GRAIL]]: 物体位姿追踪初始化

## 相关概念
- [[SAM2]]
- [[MoGe-2]]
