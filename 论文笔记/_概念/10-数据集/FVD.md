---
type: concept
aliases: [Frechet Video Distance, 弗雷歇视频距离]
---

# FVD

## 定义
衡量生成视频与真实视频分布之间距离的指标，基于 I3D 特征空间中的 Frechet 距离计算。

## 数学形式

$$
\text{FVD} = \lVert \mu_r - \mu_g \rVert^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

## 核心要点
1. 值越低表示生成视频质量越好
2. 类似图像领域的 FID，但使用视频特征提取器
3. 是视频生成领域最常用的定量指标

## 代表工作
- [[MultiWorld]]: 使用 FVD 作为主要评估指标

## 相关概念
- [[LPIPS]]
- [[SSIM]]
- [[PSNR]]
