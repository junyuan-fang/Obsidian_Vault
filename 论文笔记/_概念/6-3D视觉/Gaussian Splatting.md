---
type: concept
aliases: [3DGS, Gaussian Splatting, 高斯泼溅]
---

# Gaussian Splatting

## 定义

用大量带不透明度/颜色/协方差的 3D 高斯椭球显式表示场景，通过可微光栅化渲染图像的新视图合成方法。相较 [[NeRF]] 在训练/推理速度上有数量级提升，已成为 [[Neural Reconstruction|神经重建]] 模拟器的常用底层。

## 数学形式

$$
C(\mathbf{x}) = \sum_{i} c_i \,\alpha_i \prod_{j<i}(1-\alpha_j),\quad \alpha_i = o_i \cdot \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right)
$$

## 核心要点

1. 场景 = 一群可学习 3D 高斯
2. 可微光栅化使监督容易
3. 实时渲染、训练快
4. 是当前驾驶 NuRec / 城市级重建主流之一

## 代表工作

- 3D Gaussian Splatting (SIGGRAPH 2023)
- 在 [[OmniDreams]] 论文中作为 [[Neural Reconstruction|NuRec]] 的代表方法

## 相关概念

- [[3D Gaussian Splatting]]
- [[NeRF]]
- [[Neural Reconstruction]]
