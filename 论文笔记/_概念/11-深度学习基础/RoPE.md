---
type: concept
aliases: [Rotary Position Embedding, 旋转位置编码]
---

# RoPE

## 定义
通过旋转矩阵将位置信息编码到 Query/Key 向量中的位置编码方法，使注意力计算天然包含相对位置信息。

## 数学形式

$$
\text{RoPE}(\mathbf{x}, m) = \mathbf{R}_m \mathbf{x}, \quad \mathbf{R}_m = \begin{bmatrix} \cos(m\theta_1) & -\sin(m\theta_1) \\ \sin(m\theta_1) & \cos(m\theta_1) \\ & & \ddots \end{bmatrix}
$$

## 核心要点
1. 旋转编码在点积中自然退化为相对位置：$(\mathbf{R}_m \mathbf{q})^T (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$
2. 基频 $\theta_j = b^{-2j/D}$ 控制不同维度的旋转速度
3. 可泛化到任意长度，无需重新训练

## 代表工作
- [[MultiWorld]]: 创新性地将 RoPE 用于 Agent Identity Embedding（$b=20$）

## 相关概念
- [[Attention]]
- [[Transformer]]
