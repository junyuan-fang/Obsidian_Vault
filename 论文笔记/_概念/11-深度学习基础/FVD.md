---
type: concept
aliases: [FVD, Frechet Video Distance]
---

# FVD

## 定义
视频生成质量度量，类似 FID 但作用在 I3D 等视频特征空间，比较生成视频与真实视频的 Frechet 距离。

## 数学形式
$$
$\mathrm{FVD} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$
$$

## 核心要点
1. 特征通常来自 I3D-Inflated-3D-ConvNet
2. 对时序一致性敏感

## 代表工作
- [[minWM]]: 讨论中作为未公开的潜在指标

## 相关概念
- [[Video Diffusion Model]]
