---
type: concept
aliases: [Wan2.1, Wan 2.1, Wan-2.1, 万相 2.1]
---

# Wan2.1

## 定义

Wan2.1 是阿里通义"万相"系列的开源视频扩散基座（14B 参数为主版本），支持文本/图像到视频生成，使用 spatial-temporal VAE + DiT 结构。在机器人 / 世界模型社区被广泛当做 finetune 基座，因为开源权重、VAE 质量高、支持长时序生成。

## 核心要点

1. 14B / 1.3B 多档参数。
2. 自带视频 VAE，spatial-temporal latent，便于 latent video diffusion 工作流。
3. 支持 image-to-video（I2V），天然契合"以当前观测为起点生成未来"的世界模型范式。
4. 微调常走 LoRA，显存友好。

## 代表工作

- [[PiL-World]]：在 Wan2.1-14B 上 LoRA 微调，做多视角 chunk-wise 机器人世界模型。
- 多个最近的 video world model / VLA video prior 工作都以 Wan 系列为基座。

## 相关概念

- [[Video Diffusion Model]]
- [[Video Foundation Model]]
- [[LoRA]]
- [[VAE]]
- [[Cosmos]]
