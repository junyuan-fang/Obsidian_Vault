---
type: concept
aliases: [VideoActionDiT, Video-Action Diffusion Transformer]
---

# Video-Action DiT

## 定义

把 [[Video Diffusion Model|视频扩散]] 头（VideoDiT $\mathcal{M}_\theta$）和动作扩散头（ActionDiT $\pi_\phi$）**共享同一 [[DiT|Diffusion Transformer]] 主干、用 [[Flow Matching]] 联合训练**的 [[World-Action Model|WAM]] 实现形态。视觉侧学世界动力学，动作侧学策略，二者通过共享表征互利。

## 数学形式

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}\Big[\,\|v_\theta(z_\tau,\cdot) - (z_1-z_0)\|^2 + \|v_\phi(a_\tau,\cdot) - (a_1-a_0)\|^2\,\Big]
$$

其中 $z_\tau = \tau z_1 + (1-\tau) z_0$、$a_\tau = \tau a_1 + (1-\tau) a_0$，$\tau \sim \mathcal{U}(0,1)$。

## 核心要点

1. **共享 backbone**: 视觉和动作两个 head 共享同一 [[DiT]] transformer，参数效率高
2. **训练目标**: 双侧都用 [[Flow Matching]]，把噪声端线性流到数据端
3. **典型规模**: [[AdaWAM]] 中 VideoDiT 基于 [[Wan2.1|Wan2.2]]-5B 蒸馏，ActionDiT 是 1B 压缩版（hidden 1024）
4. **推理灵活**: VideoDiT 可被路由器关掉（$\langle\text{VR}\rangle=0$），只让 ActionDiT 出 action chunk
5. **优于纯 VLA**: 视觉预测充当"想象先验"，长程任务上明显优于 action-only

## 代表工作

- [[AdaWAM]]: 共享 backbone + 路由稀疏激活
- Fast-WAM / Motus: video-action joint 但 *每步都做梦*，无路由

## 相关概念

- [[World-Action Model]]
- [[DiT]]
- [[Flow Matching]]
- [[Video Diffusion Model]]
- [[Wan2.1]]
