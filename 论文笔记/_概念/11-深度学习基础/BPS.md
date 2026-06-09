---
type: concept
aliases: [Basis Point Set]
---

# BPS (Basis Point Set)

## 定义
一组预定义的固定 basis 点（通常在 unit ball 内随机采样数百个点），通过编码 mesh / 点云到这些 basis 点的最近距离向量来获得固定维度的形状描述子。

## 数学形式
$$
\mathrm{BPS}(\mathcal{O}) = \big[\min_{p\in\mathcal{O}}\|b_i - p\|\big]_{i=1}^N
$$

## 核心要点
1. **固定维度**：与点云规模无关；
2. **平移 / 旋转敏感**：通常先标准化；
3. **GRAIL 用法**：作为物体形状条件输入 [[Object-Aware Adaptor]]。

## 代表工作
- Prokudin et al. 2019
- [[GRAIL]]

## 相关概念
- [[Object-Aware Adaptor]]
