---
type: concept
aliases: [Finite Scalar Quantization]
---

# FSQ (Finite Scalar Quantization)

## 定义
有限标量量化，是 VQ-VAE 的替代方案：把每一维 latent 独立映射到一个固定级数的量化网格，无需 codebook 也无需 commitment loss。

## 数学形式
对每一维 $z_d$：$\hat z_d = \mathrm{round}(z_d) / L$，其中 $L$ 是每维量化级数。

## 核心要点
1. **简化**：无 codebook collapse 问题；
2. **训练稳定**：straight-through 梯度即可；
3. **应用**：[[SONIC]] 用 FSQ 量化全身控制 latent。

## 代表工作
- Mentzer et al. 2024
- [[SONIC]], [[GRAIL]]

## 相关概念
- [[SONIC]]
- [[Object-Aware Adaptor]]
