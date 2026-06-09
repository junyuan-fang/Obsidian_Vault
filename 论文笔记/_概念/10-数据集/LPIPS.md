---
type: concept
aliases: [Learned Perceptual Image Patch Similarity, 感知相似度]
---

# LPIPS

## 定义
基于深度网络特征的感知相似度指标，通过比较图像在预训练网络中间层的特征距离来评估感知质量。

## 核心要点
1. 值越低表示感知相似度越高
2. 比 PSNR/SSIM 更符合人类视觉感知
3. 常用 VGG 或 AlexNet 作为特征提取器

## 代表工作
- [[MultiWorld]]: 作为视频生成质量评估指标之一

## 相关概念
- [[FVD]]
- [[SSIM]]
- [[PSNR]]
