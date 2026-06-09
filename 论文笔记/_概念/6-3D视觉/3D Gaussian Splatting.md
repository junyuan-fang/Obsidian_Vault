---
type: concept
aliases: [3DGS, Gaussian Splatting]
---

# 3D Gaussian Splatting

## 定义

Kerbl et al., 2023 提出的显式 3D 表征：用大量 3D 各向异性高斯（位置 $\mu$、协方差 $\Sigma$、颜色 + 不透明度）拟合场景，通过可微分溅射光栅化实时渲染照片级图像。

## 数学形式

每个高斯被光栅化为屏幕上的 2D 高斯：

$$
\Sigma' = J W \Sigma W^\top J^\top
$$

$W$ 为视图变换、$J$ 为投影雅可比。最终像素 $C = \sum_i T_i \alpha_i c_i$，按深度顺序 alpha-blend。

## 核心要点

1. **实时渲染**：纯前向光栅化，1080p 可达 100+ FPS
2. **可微**：可端到端优化参数
3. **几何与外观分离**：相比 mesh 更柔性，相比 NeRF 更快
4. **在具身领域**：天然适配 sim-to-real，因为渲染图与真实照片几乎不可区分

## 代表工作

- Kerbl et al., 2023: 原始 3DGS
- [[GaussNav]] / [[SAGE-Bench]] / [[GN0]]: 用 3DGS 做导航数据/仿真
- [[3DGS Dynamic Avatar]]: 动态扩展
- 4D-GS / Deformable-GS: 动态场景扩展

## 相关概念

- [[InteriorGS]]
- [[diff-gaussian-rasterization]]
- [[BEV Memory]]
- [[3DGS Dynamic Avatar]]
