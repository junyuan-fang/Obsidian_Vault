---
type: concept
aliases: [深度对齐损失]
---

# Depth Alignment Loss

## 定义
利用 [[MoGe-2]] 估计的 metric 深度反投得到的点云，与人体 / 物体 mesh 可见顶点之间的 [[Chamfer Distance|Chamfer 距离]] 损失，约束 3D 几何与真实物理尺度一致。

## 数学形式
$$
\mathcal{L}_{depth} = \frac{1}{T}\sum_{t=1}^{T}\big[\mathrm{CD}(V^{H,vis}_t, P^H_t) + \mathrm{CD}(V^{O,vis}_t, P^O_t)\big]
$$

## 核心要点
1. **metric scale 恢复**：依赖背景深度标定；
2. **仅可见顶点**：避免被遮挡部分误对齐；
3. **GRAIL 消融**：去掉后下游追踪 SR 从 81.4% → 42.6%。

## 代表工作
- [[GRAIL]]

## 相关概念
- [[GRAIL Reconstruction Loss]]
- [[MoGe-2]]
- [[Chamfer Distance]]
