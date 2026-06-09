---
type: concept
aliases: [Peak Signal-to-Noise Ratio, 峰值信噪比]
---

# PSNR

## 定义
基于像素级均方误差的图像质量指标，值越高表示重建质量越好。

## 数学形式

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

## 核心要点
1. 单位为 dB，值越高越好
2. 计算简单但不完全反映感知质量
3. 常与 SSIM、LPIPS 配合使用

## 相关概念
- [[SSIM]]
- [[LPIPS]]
- [[FVD]]
