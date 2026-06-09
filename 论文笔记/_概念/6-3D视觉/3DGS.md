---
type: concept
aliases: [3D Gaussian Splatting, 三维高斯泼溅]
---

# 3DGS

## 定义
3D Gaussian Splatting 是一种显式 3D 场景表示方法，用各向异性 3D 高斯基元表示场景，支持实时可微渲染。

## 数学形式
$$G(x) = e^{-rac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)}$$

## 核心要点
1. 每个 Gaussian 有位置 $\mu$、协方差 $\Sigma$、球谐系数（颜色）和不透明度
2. 通过 splatting（投影+alpha blending）实现实时渲染
3. 支持自适应密度控制（split/clone/prune）

## 代表工作
- [[3D Gaussian Splatting]]: Kerbl et al. 2023, 原始论文
- [[Mip-Splatting]]: 抗锯齿改进

## 相关概念
- [[NeRF]]