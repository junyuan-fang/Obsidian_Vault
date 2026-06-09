---
type: concept
aliases: [Latent History Memory, 潜空间历史记忆, History Latent Conditioning]
---

# Latent History Memory

## 定义

在视频 / 世界模型生成的每一步（或每一轮闭环 rollout）中，把最近若干帧用 VAE 编码成 latent token，作为生成下一段未来帧的条件，从而保持跨步 / 跨轮的视觉一致性（物体位置、光照、姿态）。与简单的"上一帧条件"相比，它显式维护一个**多帧记忆窗**。

## 数学形式

$$
\mathcal{H}_t = \{Z_t^{v,h}\}_{v\in\mathcal{V}}, \quad Z_t^{v,h} = E_\phi(\{x_\tau^v\}_{\tau \in \mathcal{I}_t^h})
$$

其中 $\mathcal{I}_t^h$ 是当前时刻往前 $H_h$ 帧的索引集合，$E_\phi$ 是视频 VAE 编码器。

## 核心要点

1. 跨轮一致性的关键：去掉它会导致长 rollout 物体凭空挪动 / 手臂跳变。
2. latent 形式让"记忆"代价远低于像素级历史 buffer。
3. 滑窗更新：每轮把新生成帧并入，移除最老帧。
4. 在 PiL-World 消融中，去掉 latent history memory 让 LPIPS 恶化 2.5–3.3 倍。

## 代表工作

- [[PiL-World]]：把它作为 4 大核心模块之一，与 [[Action-to-Control Projection]] 配对。

## 相关概念

- [[VAE]]
- [[Video Diffusion Model]]
- [[Causal Attention]]
- [[Exposure Bias]]
