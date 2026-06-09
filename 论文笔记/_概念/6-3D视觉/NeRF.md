---
type: concept
aliases: [Neural Radiance Field, 神经辐射场]
---

# NeRF

## 定义
Neural Radiance Field 用 MLP 将 3D 坐标和视角方向映射为颜色和密度，通过体渲染合成新视角。

## 数学形式
$$C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt$$

## 核心要点
1. 隐式场景表示，连续且紧凑
2. 需要密集的多视角图像训练
3. 渲染速度慢是主要瓶颈，3DGS 等方法改进了这点

## 相关概念
- [[3DGS]]