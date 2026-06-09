---
type: concept
aliases: [LPIPS, Learned Perceptual Image Patch Similarity, 感知相似度]
---

# LPIPS (Learned Perceptual Image Patch Similarity)

## 定义

一种基于深度网络中间特征的图像感知相似度指标。给定两张图像，分别通过预训练 CNN（通常是 AlexNet / VGG）提取多层特征，按通道做归一化后计算加权 L2 距离，最后跨层求和。比 PSNR / SSIM 更贴合人类对"看起来像不像"的判断。

## 数学形式

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{f}_l^{x}(h,w) - \hat{f}_l^{y}(h,w)) \|_2^2
$$

- $\hat{f}_l^{x}, \hat{f}_l^{y}$: 第 $l$ 层归一化后的特征
- $w_l$: 各通道学习到的权重
- $H_l, W_l$: 该层特征图尺寸

## 核心要点

1. 越小越像，0 表示完全一致。
2. 默认 AlexNet backbone 计算快，VGG backbone 更准。
3. 是 video generation / 世界模型评测的标准 visual fidelity 指标。
4. 对全局色调 / 纹理变化敏感，但对几何错位的惩罚弱于 SSIM 在某些 case 下。

## 代表工作

- [[PiL-World]]：用 LPIPS 衡量单步多视角预测保真度，head-view 上比 [[Ctrl-World]] 降幅 −38% ~ −42%。

## 相关概念

- [[KL Divergence]]
- [[FVD]]
- [[Video Diffusion Model]]
