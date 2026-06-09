---
type: concept
aliases: [Hand model]
---

# MANO

## 定义
MANO 是 SMPL 系列的手部参数化模型，提供手部 shape + pose 的低维表示（pose 维度约 45）。

## 核心要点
1. 顶点 ~778 / 手；
2. 关节 16；
3. 与 [[SMPL-X]] 无缝衔接（wrist 对齐 + 手指扩展）；
4. RGB 单视图手部估计的事实输出格式。

## 代表工作
- Romero et al. 2017
- [[WiLoR]], [[GRAIL]]

## 相关概念
- [[SMPL-X]]
- [[Inverse Kinematics]]
