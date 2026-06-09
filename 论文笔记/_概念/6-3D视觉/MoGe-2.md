---
type: concept
aliases: [Monocular Geometry Estimation]
---

# MoGe-2

## 定义
单目 metric 深度 + 几何估计模型，从 RGB 图像预测度量尺度深度图和法线 / 几何属性。

## 核心要点
1. **metric depth**：直接输出米制深度（非相对）；
2. 是 [[GRAIL]] 深度对齐损失的关键先验；
3. 配合已知背景深度可恢复全场景尺度。

## 代表工作
- Wang et al. 2025
- [[GRAIL]]

## 相关概念
- [[Metric Depth]]
- [[Chamfer Distance]]
