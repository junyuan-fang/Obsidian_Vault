---
type: concept
aliases: [MMDiT, Multi-Modal Diffusion Transformer]
---

# MMDiT

## 定义
Stable Diffusion 3 / HunyuanVideo 系列采用的扩散 Transformer 架构，文本与图像/视频 token 在共享 attention 里联合处理。

## 数学形式
$$
$[T; V] \to \text{SelfAttn}([T; V]) \to [\hat T; \hat V]$
$$

## 核心要点
1. 文本与视觉 token 拼接做 self-attention
2. 相比 cross-attention DiT 更适合多模态对齐
3. HunyuanVideo / SD3 / FLUX 使用

## 代表工作
- [[minWM]]: HY1.5-TI2V-8B 即 MMDiT 骨干，验证 PRoPE 通用性

## 相关概念
- [[Self-Attention]]
- [[Video Diffusion Model]]
