---
type: concept
aliases: [Diffusion Transformer, diffusion-transformer]
---

# DiT (Diffusion Transformer)

## 定义

Diffusion Transformer：把 [[Diffusion Model]] 的 U-Net backbone 替换为 Transformer 的架构（Peebles & Xie, 2022），后续成为 [[Flow Matching]] / 视频生成 / [[VLA]] action head 的事实标准 backbone（[[MMDiT]] 是 multimodal 扩展、[[AdaLN]] 是常用 conditioning 方式）。

## 核心要点

1. **结构**: patchify → Transformer block × N → unpatchify，timestep & condition 通过 [[AdaLN]] 注入
2. **优势**: 比 U-Net 更易 scale，适合 long-context 与多模态条件
3. **action head 应用**: [[Pi05|π₀.₅]] / [[LARA]] 等 VLA 把 DiT 当 action expert，输出 [[Action Chunking|动作 chunk]] 的 [[Flow Matching]] velocity
4. **cross-attention vs in-context**: 条件可以 cross-attn 进来（[[LARA]]）也可以拼接成 in-context token（MMDiT）

## 代表工作

- Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)
- Stable Diffusion 3 / FLUX / [[Wan2.1]]: 视频生成
- [[Pi05]] / [[LARA]]: VLA action head

## 相关概念

- [[Diffusion Model]]
- [[Flow Matching]]
- [[MMDiT]]
- [[AdaLN]]
