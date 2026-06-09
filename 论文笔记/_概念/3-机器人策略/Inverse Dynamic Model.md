---
type: concept
aliases: [IDM, 逆动力学模型, inverse-dynamics]
---

# Inverse Dynamic Model (IDM)

## 定义

Inverse Dynamic Model：给定两帧观测 $(I_t, I_{t+C})$，反推出"让 $I_t$ 演化到 $I_{t+C}$ 所需的动作"。在 [[Latent Action Model|LAM]] 体系里 IDM 输出是 latent action $z_t$，在传统 robotics 里 IDM 输出真实控制量。

## 数学形式

$$
z_t = \Phi_\text{IDM}(I_t, I_{t+C}) \quad \text{or} \quad a_t = \Phi_\text{IDM}(s_t, s_{t+1})
$$

## 核心要点

1. **角色**: action 编码器，从视觉变化抽 action 信息
2. **与 FDM 对偶**: IDM 抽 action，[[Forward Dynamic Model|FDM]] 用 action 预测未来 → 两者构成 VQ-VAE 风格的 encoder-decoder
3. **训练信号**: 像素重建 + commitment loss（[[Vector Quantization|VQ-VAE]] 目标）
4. **失败模式**: 容易吸收光影 / 背景变化，需要 [[Latent Action Representation Alignment|LARA]] 这种"动作监督回流"机制纠偏

## 代表工作

- [[LARA]]: 用 alignment loss 让 IDM 关注 end-effector 区域（见 LARA Figure 5 attention map）
- 经典 VPT / GENIE / LAPA 中的 IDM 模块

## 相关概念

- [[Forward Dynamic Model]]
- [[Latent Action Model]]
- [[Vector Quantization]]
