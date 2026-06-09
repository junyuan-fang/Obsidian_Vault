---
type: concept
aliases: [重投影误差, RPE]
---

# Reprojection Error

## 定义
多视角几何中衡量 3D 重建精度的指标，计算 3D 点投影到图像平面后与实际观测位置的偏差。

## 数学形式

$$
\text{RPE} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \lVert p^*_{ij} - \Pi(P_{ij}) \rVert_2
$$

## 核心要点
1. 值越低表示多视角间几何一致性越好
2. 需要先进行特征匹配和三角化
3. 是评估多视角生成一致性的关键指标

## 代表工作
- [[MultiWorld]]: 使用 RPE 评估多视角视频生成的几何一致性

## 相关概念
- [[VGGT]]
