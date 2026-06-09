---
type: concept
aliases: [Video Diffusion Model, 视频扩散模型]
---

# Video Diffusion Model

## 定义
在视频 latent 空间上做去噪扩散的生成模型，骨干通常是 3D U-Net 或 video DiT，同时建模空间与时间维度。

## 数学形式
$$
$x_0 \sim p_{data}(x_0)$, $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
$$

## 核心要点
1. 典型骨干：MMDiT、cross-attention DiT、3D U-Net
2. 条件信号：文本、图像、相机轨迹、动作 token
3. 训练目标通常是 v-prediction 或 epsilon-prediction

## 代表工作
- [[minWM]]: 把双向 video diffusion 蒸馏为实时自回归世界模型

## 相关概念
- [[Diffusion Model]]
- [[MMDiT]]
- [[Video World Model]]
