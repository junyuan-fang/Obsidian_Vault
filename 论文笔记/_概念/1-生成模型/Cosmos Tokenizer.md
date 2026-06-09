---
type: concept
aliases: [Cosmos VAE, Cosmos Tokenizer]
---

# Cosmos Tokenizer

## 定义

NVIDIA [[Cosmos]] 配套的时空压缩 tokenizer：把高分辨率视频压成 latent patch token，供 [[Video Diffusion Model|视频扩散]] / [[Mixture-of-Transformers|MoT]] backbone 处理，是 Cosmos 系列复用的视觉前端。

## 核心要点

1. **时空联合压缩**: 时间×空间双轴下采样，典型 8×8 空间 + 4× 时间
2. **连续 latent + 离散 token 双形态**: 连续给 diffusion 用，离散给 AR 用
3. **重建保真高**: 在 PSNR / SSIM / FVD 上优于通用 SD-VAE
4. **训练数据规模**: 与 Cosmos backbone 同源,数千万小时视频

## 代表工作

- [[Cosmos]]: 引入原版 Cosmos Tokenizer
- [[Cosmos3]]: 继承并扩展至音频、动作 token
- [[OmniDreams]]: 直接复用 Cosmos Tokenizer 做驾驶视频压缩

## 相关概念

- [[Cosmos]]
- [[Video Diffusion Model]]
- [[Trajectory Tokenization]]
