---
type: concept
aliases: [Differentiable Gaussian Rasterization]
---

# diff-gaussian-rasterization

## 定义

[[3D Gaussian Splatting|3DGS]] 原始论文（Kerbl et al., 2023）发布的可微分高斯溅射 CUDA 光栅化后端，是绝大多数 3DGS 工作的渲染基座。

## 核心要点

1. **CUDA tile-based**：把屏幕分块，每块独立 alpha-blend
2. **可微**：颜色、不透明度、协方差全部回传梯度
3. **支持 RGB-D + 自定义投影**：可注入正交投影矩阵渲染 BEV
4. **典型扩展**：4D 时间维 / 各向异性正则 / 法线监督

## 代表工作

- Kerbl et al., 2023: 原始版本
- [[GN0]]: 在 GN-Bench 中作为核心渲染引擎，同时输出 FPV 与 BEV

## 相关概念

- [[3D Gaussian Splatting]]
- [[BEV Memory]]
