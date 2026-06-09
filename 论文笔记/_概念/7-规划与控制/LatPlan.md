---
type: concept
aliases: [Latent Planning, AMA3, LatPlan-AMA3]
---

# LatPlan

## 定义

Asai et al. 系列工作（2018–2022），首批尝试**从图像直接学经典符号规划模型**的神经符号方法。代表版本 **AMA3** 端到端学：

1. **Binary latent VAE**: 把图像编码为 $m$ 维 0/1 latent code（用 Gumbel-softmax / discrete VAE）
2. **Action-conditioned neural transition**: 神经网络 $\phi(z_t, a_t) \to z_{t+1}$
3. **STRIPS-style precondition / effect mask**: 直接从转移数据回归得到

测试时用经典 STRIPS 规划器（FF, LAMA）在 $m$ 位 latent 上搜索。

## 与 STRIPS-WM 对比

| 维度 | LatPlan-AMA3 | [[STRIPS-WM]] |
|---|---|---|
| 状态编码 | 端到端 binary VAE | [[FSQ]] + [[JEPA]] 自监督 |
| 算子学习 | 神经网络回归 | [[CP-SAT]] 布尔约束 |
| 正确性 | 无形式保证 | $\xi=\eta=0$ 时 sound & complete |
| 长 horizon | 显著退化 | 100% 成功 |

## 代表工作

- Asai & Fukunaga 2018: "Classical Planning in Deep Latent Space"
- Asai et al. 2022: AMA3, JAIR 74

## 相关概念

- [[STRIPS]]
- [[Propositional Planning]]
- [[STRIPS-WM]]
