---
type: concept
aliases: [Diffusion Model, 扩散模型, DDPM]
---

# Diffusion Model

## 定义
通过马尔可夫前向加噪 + 反向去噪学习数据分布的生成模型，本质是学习 score function。

## 数学形式
$$
$dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dW$ (forward SDE)
$$

## 核心要点
1. 前向：高斯加噪到 $\mathcal{N}(0,I)$
2. 反向：参数化为 score $\nabla_x \log p_t(x)$ 或 epsilon
3. 采样：DDPM / DDIM / DPM-Solver / Flow Matching

## 代表工作
- [[minWM]]: 作为基础生成范式

## 相关概念
- [[Score Function]]
- [[Video Diffusion Model]]
