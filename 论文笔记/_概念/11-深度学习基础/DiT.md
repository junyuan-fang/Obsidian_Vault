---
type: concept
aliases: [Diffusion Transformer]
---

# DiT

## 定义
Diffusion Transformer 将 Transformer 架构引入扩散模型，替代传统 U-Net backbone，在图像/视频生成中取得 SOTA。

## 核心要点
1. 用 patchify 将图像转为 token 序列
2. 通过 adaptive layer norm (AdaLN) 注入时间步和条件信息
3. 规模效应明显，更大模型 = 更好 FID

## 代表工作
- Peebles & Xie, 2023: 原始 DiT 论文

## 相关概念
- [[Transformer]]
- [[DDPM]]