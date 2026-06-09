---
type: concept
aliases: [多任务损失, Multi-Task Learning Loss]
---

# Multi-Task Loss

## 定义
将多个任务的损失函数通过加权求和组合为统一的训练目标，使模型同时学习多个相关任务。

## 数学形式

$$
\mathcal{L}_{total} = \sum_i \lambda_i \mathcal{L}_i
$$

## 核心要点
1. 权重 $\lambda_i$ 的选择对性能至关重要
2. 辅助任务可提供正则化效果和额外监督信号
3. 常见策略：手动调参、不确定性加权、梯度平衡

## 代表工作
- [[GigaWorld-Policy]]: $\lambda_{action}=5, \lambda_{video}=1$，强调动作预测同时保留视频一致性正则化

## 相关概念
- [[Flow Matching]]
