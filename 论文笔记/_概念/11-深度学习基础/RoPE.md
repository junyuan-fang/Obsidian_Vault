---
type: concept
aliases: [RoPE, Rotary Position Embedding, 旋转位置编码]
---

# RoPE

## 定义
用旋转矩阵把绝对位置编码进 Q/K，使 attention 仅依赖相对位置。

## 数学形式
$$
$\text{RoPE}(x_m, m) = R_m x_m,\quad \langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$
$$

## 核心要点
1. 最早 Su et al. 2021 提出（RoFormer）
2. 适合长序列外推
3. 可推广到 2D / 3D / 相机几何（如 PRoPE）

## 代表工作
- [[minWM]]: PRoPE 的母体位置编码

## 相关概念
- [[Self-Attention]]
- [[PRoPE]]
