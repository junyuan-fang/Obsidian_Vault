---
type: concept
aliases: [Neural Radiance Field, 神经辐射场]
---

# NeRF

## 定义

用 MLP 隐式表示场景的辐射场 $F_\theta : (\mathbf{x}, \mathbf{d}) \mapsto (c, \sigma)$，通过体积渲染合成新视角图像的方法。

## 数学形式

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,c(\mathbf{r}(t), \mathbf{d})\,dt,\quad T(t)=\exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)
$$

## 核心要点

1. 隐式连续场景表示
2. 训练慢、渲染慢
3. 被 [[Gaussian Splatting]] 在驾驶/室外重建任务上大量替代
4. 仍是 [[Neural Reconstruction|神经重建]] 模拟器思路的起点

## 代表工作

- NeRF (ECCV 2020)
- 后续 Instant-NGP、Mip-NeRF 等加速版

## 相关概念

- [[Gaussian Splatting]]
- [[Neural Reconstruction]]
