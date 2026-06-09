---
type: concept
aliases: [T2I, 文生图]
---

# Text-to-Image

## 定义

输入自然语言、输出图像的生成任务,通常用 [[Diffusion Model|扩散模型]] / [[Flow Matching|flow matching]] 实现,代表系列 SD / DALL·E / Imagen / Flux / [[Cosmos3]]-T2I。Artificial Analysis 是常用第三方排行榜。

## 核心要点

1. **文本编码**: 早期 CLIP / T5，新一代用 LLM hidden state 作为条件
2. **生成主干**: U-Net → DiT → MMDiT → MoT 演进
3. **评测**: FID / CLIPScore / 用户偏好（Artificial Analysis Arena）
4. **后训练**: SFT + DPO / [[DMD|distribution matching distillation]] 蒸馏到 few-step

## 代表工作

- SD3 / Flux：开源 SOTA 基线
- DALL·E 3 / Imagen 3：闭源前沿
- [[Cosmos3]]-Super-Text2Image：Artificial Analysis 开源第一（2026）

## 相关概念

- [[Diffusion Model]]
- [[MMDiT]]
- [[Flow Matching]]
