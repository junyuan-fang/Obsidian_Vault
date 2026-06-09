---
type: concept
aliases: [FDM, 前向动力学模型, forward-dynamics]
---

# Forward Dynamic Model (FDM)

## 定义

Forward Dynamic Model：给定当前观测 $I_t$ 与动作 / latent action $z_t$，预测未来观测 $\hat I_{t+C}$。是 [[Latent Action Model|LAM]] 体系中的 *decoder*，与 [[Inverse Dynamic Model|IDM]] 共同构成 encoder-decoder。

## 数学形式

$$
\hat I_{t+C} = \Phi_\text{FDM}(I_t, z_t^q)
$$

## 核心要点

1. **角色**: action / latent 解码器，把动作信息反映射回未来视觉
2. **训练信号**: $\|I_{t+C} - \hat I_{t+C}\|_2^2$ 像素重建（也可加 [[LPIPS]] / [[FVD]] perceptual）
3. **隐式约束**: 强制 IDM 的 latent 必须 *能预测未来* → 防止 latent 退化成 trivial code
4. **关联**: FDM 给下游 VLA 注入"动作执行后视觉会变成什么样"的隐式先验，[[LARA]] 通过 alignment loss 把这一先验传给 [[DiT]]

## 代表工作

- [[LARA]]: FDM 反向正则化 VLA，减少功能上不可达的轨迹
- World Model 系列（[[World Model]] / [[Video World Model]]）本质都是 FDM 的扩展

## 相关概念

- [[Inverse Dynamic Model]]
- [[Latent Action Model]]
- [[World Model]]
