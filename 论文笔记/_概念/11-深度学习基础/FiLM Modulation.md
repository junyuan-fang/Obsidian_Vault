---
type: concept
aliases: [FiLM, Feature-wise Linear Modulation]
---

# FiLM Modulation

## 定义

**Feature-wise Linear Modulation** —— 用条件 $c$ 生成 scale $\gamma(c)$ 与 shift $\beta(c)$,对中间特征 $F$ 做逐通道仿射:

$$
\mathrm{FiLM}(F,\ c) = \gamma(c) \odot F + \beta(c)
$$

是 [[AdaLN]] 的前身/姊妹,在 visual reasoning、conditional generation、policy conditioning 中广泛使用。

## 与 [[AdaLN]] 的区别

| 维度 | FiLM | [[AdaLN]] |
|------|------|-----------|
| 应用位置 | conv/feature map | transformer LayerNorm 之后 |
| 是否含 LN | 否（直接乘加） | 是（先 LN 再乘加） |
| 起源 | Perez 2017 (FiLM) | DiT (Peebles & Xie 2022) |
| 典型用途 | conditional CNN / policy embedding | conditional transformer / 扩散 |

## 在 [[DexFuture]] 中的角色

per-link visuomotor token 构造时,用 hand link 类型 / 手别 ID 通过 FiLM 调制 local cross-attention 输出,使同一 vision backbone 对不同 link 产生区分化嵌入。

## 代表工作

- FiLM (Perez et al. 2017): visual reasoning
- BC-Z / RT-1: policy conditioning
- [[DexFuture]]: per-link token 调制

## 相关概念

- [[AdaLN]]
- [[Horizon-Conditioned Transformer]]
