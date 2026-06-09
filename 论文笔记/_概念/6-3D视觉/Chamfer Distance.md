---
type: concept
aliases: [CD, Chamfer 距离]
---

# Chamfer Distance

## 定义
两组点云 $A, B$ 之间的双向最近邻距离平均，常用于点云对齐 / 重建质量度量。

## 数学形式
$$
\mathrm{CD}(A, B) = \frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\|a-b\|^2 + \frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\|a-b\|^2
$$

## 核心要点
1. **对应无关**：不需点-点对应；
2. **可导**：在可微渲染 / 点云优化中广泛使用；
3. **变体**：单向 CD、z 方向 CD（接触对齐）。

## 代表工作
- 经典 3D 重建度量
- [[GRAIL]]: 深度对齐与接触对齐损失里都用

## 相关概念
- [[MoGe-2]]
- [[Metric Depth]]
