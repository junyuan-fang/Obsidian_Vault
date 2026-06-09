---
type: concept
aliases: [扩散策略]
---

# Diffusion Policy

## 定义

Chi et al., 2023 提出的视觉运动策略范式：把动作序列建模为条件扩散过程，由网络迭代去噪生成动作块（action chunk）。

## 数学形式

加噪与去噪目标：

$$
a_\tau = \sqrt{\bar\alpha_\tau}\, a + \sqrt{1-\bar\alpha_\tau}\, \epsilon, \quad \mathcal{L} = \mathbb{E}\bigl[\|\hat\epsilon_\theta(a_\tau, \tau, o) - \epsilon\|_2^2\bigr]
$$

## 核心要点

1. **优点**：多模态动作分布、强抗噪、对 receding-horizon 友好
2. **典型输出**：action chunk（如未来 8/16 步）
3. **常见骨干**：CNN-Unet / Transformer-DiT
4. **在导航中**：用作 Action Expert 平滑底层动作（[[NavDP]]）

## 代表工作

- Chi et al., 2023: 原始 Diffusion Policy
- [[NavDP]]: 导航扩展
- [[GN0]]: 把 NavDP 作为 BAE 的 Action Expert
- π₀ / π0.5: 大规模 VLA 也用扩散头

## 相关概念

- [[DDPM]]
- [[NavDP]]
