---
type: concept
aliases: [Joint Embedding Predictive Architecture, 联合嵌入预测架构]
---

# JEPA

## 定义
LeCun 提出的自监督学习框架，在 latent space 中预测未来嵌入，而非在像素空间重建。

## 核心要点
1. 避免像素级重建的高计算成本
2. 学到的表示聚焦于高层语义和结构
3. 需要防止表示坍塌（常用 EMA、正则化等技巧）

## 代表工作
- I-JEPA: 图像域 JEPA
- [[LeWM]]: 端到端稳定 JEPA

## 相关概念
- [[DINO]]