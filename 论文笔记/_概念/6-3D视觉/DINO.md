---
type: concept
aliases: [Self-DIstillation with NO labels]
---

# DINO

## 定义
Meta 的自监督视觉 Transformer 预训练方法，通过自蒸馏（teacher-student）学习强视觉特征。

## 核心要点
1. Teacher 用 EMA 更新，student 直接训练
2. DINOv2 进一步提升到 foundation model 级别
3. 特征具有涌现的语义分割能力

## 相关概念
- [[CLIP]]
- [[JEPA]]