---
type: concept
aliases: [多视角融合, Multi-View Encoding]
---

# Multi-View Fusion

## 定义

把同一时刻、不同相机视角的图像融合为统一表征的技术。常见做法：

1. **Late fusion**: 每个视角独立 encode 后 concat 或 attention 聚合
2. **Cross-view attention**: 视角间双向 attention
3. **3D-aware fusion**: 投到统一 3D 空间（如 BEV、voxel、[[3D Gaussian Splatting]]）后聚合

## 在视觉规划中的用途

单视角图像容易产生 [[Visual Aliasing]]——遮挡或视角变化导致看似相同但实际不同的状态。多视角融合通过几何不变性减少混叠。

## 与本文关系

[[STRIPS-WM]] **未采用**多视角输入，是单 RGB 输入。论文 Section 7 把"对动态背景、遮挡、多视角的泛化未知"列为局限——多视角融合是一个潜在改进方向。

## 代表工作

- BEVFormer / BEVFusion（自动驾驶）
- DUSt3R / MASt3R（3D 重建）

## 相关概念

- [[Visual Aliasing]]
- [[3D Gaussian Splatting]]
