---
type: concept
aliases: [3DGS Avatar, Animatable 3DGS Avatar]
---

# 3DGS Dynamic Avatar

## 定义

将 [[3D Gaussian Splatting|3DGS]] 表征与 SMPL-X 等参数化人体模型绑定，从而可被骨骼动画或运动捕捉数据驱动的可动 3D 化身。

## 数学形式

每帧用 6-DoF 姿态 $(R_t, T_t)$ 刚体驱动每个高斯：

$$
\mu_i^{(t)} = R_t \mu_i + T_t, \qquad \Sigma_i^{(t)} = R_t \Sigma_i R_t^\top
$$

驱动后的化身 $G^{(t)}_{avatar}$ 与静态场景 $G_{static}$ 拼接做统一光栅化。

## 核心要点

1. **流程**：单图 → 高斯重建 → SMPL-X 对齐 → 骨骼绑定 → 动作驱动
2. **动作源**：Mixamo FBX、动捕、文本生成动作
3. **作用**：让 VLN/HRI 评估从静态场景走向动态人居
4. **局限**：通常只支持刚体驱动 + 线性 blend skinning，复杂衣物/头发表现一般

## 代表工作

- [[GN0]]: 在 GN-Bench 中引入可动 3DGS 化身做 HRI 评估
- GaussianAvatar / 3DGS-Avatar: 早期单人体动态扩展

## 相关概念

- [[3D Gaussian Splatting]]
- [[Vision-Language Navigation]]
