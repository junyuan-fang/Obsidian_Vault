---
type: concept
aliases: [LAM, 潜动作模型, latent-action-model]
---

# Latent Action Model (LAM)

## 定义

Latent Action Model：从 *无 action 标签* 的视频中学习一个紧凑 latent 空间，每个 latent token 编码"两帧之间发生的动作"。结构通常是 [[Inverse Dynamic Model|IDM]]($I_t, I_{t+C}) \to z_t$ + [[Forward Dynamic Model|FDM]]($I_t, z_t) \to \hat I_{t+C}$，配 [[Vector Quantization|VQ-VAE]] 把 $z$ 离散成 codebook token。

## 数学形式

$$
z_t = \Phi_\text{IDM}(I_t, I_{t+C}), \quad \hat I_{t+C} = \Phi_\text{FDM}(I_t, z_t^q)
$$

训练目标见 [[LARA]] 公式 3。

## 核心要点

1. **目标**: 用海量无标签视频替代昂贵的真机 action 数据
2. **离散化**: VQ-VAE codebook（通常 32–256 token）把连续 latent 离散，方便下游 [[VLA]] 当 pseudo-label
3. **传统用法**: LAM 预训练完冻结，token 当作伪 action 监督 VLA（LAPA / Moto-GPT / GR00T-N1）
4. **失败模式**: latent 容易学到光影 / 镜头扰动而非控制相关变化，需要 inverse-dynamics 监督拉回

## 代表工作

- [[LARA]]: 把 LAM 与 [[VLA]] *联合优化*，跳出 frozen pseudo-label 范式
- LAPA / Moto-GPT / GR00T-N1: 传统冻结 LAM 范式

## 相关概念

- [[Inverse Dynamic Model]]
- [[Forward Dynamic Model]]
- [[Vector Quantization]]
- [[VLA]]
- [[Latent Action Representation Alignment]]
