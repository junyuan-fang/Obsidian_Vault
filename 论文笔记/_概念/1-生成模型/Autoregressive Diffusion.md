---
type: concept
aliases: [Autoregressive Diffusion, 自回归扩散, AR Diffusion]
---

# Autoregressive Diffusion

## 定义
按时间维度（或 chunk 维度）自回归地做扩散生成：每次只生成一段 latent，并以之前生成的 latent 为条件。

## 数学形式
$$
$p_\theta(x_{1:T}) = \prod_t p_\theta(x_t \mid x_{<t})$ + diffusion on each chunk
$$

## 核心要点
1. 核心：causal attention mask 切断未来视野
2. 优势：首帧低延迟、可流式生成、自然支持交互动作输入
3. 常配合 Teacher Forcing 训练

## 代表工作
- [[minWM]]: 学生模型的生成范式

## 相关概念
- [[Causal Forcing]]
- [[Causal Attention]]
- [[Teacher Forcing]]
