---
type: concept
aliases: [PRoPE, Projected Rotary Position Embedding, 投影旋转位置编码]
---

# PRoPE

## 定义
把每帧的相机内参 $K_i$ 与外参 $T_i^{cw}$ 提升为 $4\times 4$ 投影矩阵，并以 block-diagonal 方式嵌入 RoPE，使 attention 自然带有相机几何条件。

## 数学形式
$$
$\tilde P_i = \begin{bmatrix}K_i & 0 \\ T_i^{cw} & e_4^\top\end{bmatrix}$
$$

## 核心要点
1. head 维度被切分：相机块 + 2D 空间 RoPE 块
2. 无需 cross-attention 旁挂，参数零增
3. 对 MMDiT 与 cross-attention DiT 通用

## 代表工作
- [[minWM]]: 相机条件注入核心模块

## 相关概念
- [[RoPE]]
- [[Self-Attention]]
