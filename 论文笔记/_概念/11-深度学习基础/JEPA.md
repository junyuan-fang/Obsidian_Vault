---
type: concept
aliases: [Joint Embedding Predictive Architecture, I-JEPA, V-JEPA]
---

# JEPA (Joint Embedding Predictive Architecture)

## 定义

LeCun 提出的自监督表征学习范式（2022）。核心思想：**不在像素空间预测，而在 latent embedding 空间预测**。

典型结构：

- **Student 编码器** $f_\theta$: 处理当前 view / context
- **Teacher 编码器** $f_{\bar\theta}$: 由 student 的 [[EMA]] 复制，处理目标 view / future
- **Predictor** $g_\phi$: 接 student 输出（可能加 context 信息）预测 teacher target
- **损失**: 在 embedding 空间算 $\| g_\phi(z) - z^{\text{tar}} \|^2$，teacher 端 stop-gradient

## 为什么比像素重构好

| 像素重构 (e.g. MAE) | JEPA |
|---|---|
| 预测高频细节，浪费容量 | 只预测语义可预测的部分 |
| 易拟合纹理噪声 | 抽象不变量 |
| 不适合 RL / 规划 | 适合下游 latent dynamics |

## 代表工作

- I-JEPA (Assran et al. 2023): 图像
- V-JEPA (Bardes et al. 2024): 视频
- [[STRIPS-WM]] (2026): 用 JEPA + [[FSQ]] 学规划用的离散状态码

## 与本文关系

[[STRIPS-WM]] Stage 1 完全采用 JEPA：student + EMA teacher + FSQ 量化 + action-conditioned predictor + inverse dynamics 辅助。

## 相关概念

- [[EMA]]
- [[FSQ]]
- [[Inverse Dynamics]]
- [[Predictive Separation Loss]]
