---
type: concept
aliases: [Adaptive Layer Normalization, AdaLN-Zero, Conditional LayerNorm]
---

# AdaLN

## 定义

**Adaptive Layer Normalization** —— 把外部条件 $c$ 注入 LayerNorm 的 scale 与 shift,使 transformer 块的输出依赖条件信息。起源于 DiT（Diffusion Transformer）,后被广泛用于扩散模型、视频生成、条件 transformer。[[DexFuture]] 把它从 diffusion 改用于非扩散的 horizon-conditioned 回归。

## 数学形式

对 transformer 中的 LayerNorm:

$$
\mathrm{AdaLN}(Y,\ c) = \mathrm{LN}(Y) \odot (1 + s(c)) + b(c)
$$

其中:
- $\mathrm{LN}(\cdot)$: 标准 LayerNorm（去除原 affine）
- $s(c), b(c)$: 由条件 $c$ 经小 MLP 得到的 scale / shift
- $\odot$: 逐通道乘

## AdaLN-Zero 变体

DiT 提出在 residual block 中再加一个**门控 scale** $\alpha(c)$,初始化为 0,保证训练初期 identity 行为:

$$
Y \gets Y + \alpha(c) \cdot \text{block}(\mathrm{AdaLN}(Y, c))
$$

## 在 transformer block 中的常见用法

对每个 sub-layer（SA / CA / FFN）都做 AdaLN:

$$
\begin{aligned}
Y &\gets Y + \alpha_{sa}(c) \cdot \mathrm{MSA}(\mathrm{AdaLN}(Y, c)) \\
Y &\gets Y + \alpha_{ca}(c) \cdot \mathrm{MCA}(\mathrm{AdaLN}(Y, c),\ M) \\
Y &\gets Y + \alpha_{ff}(c) \cdot \mathrm{FFN}(\mathrm{AdaLN}(Y, c))
\end{aligned}
$$

## 在 [[DexFuture]] 中的角色

$c = c_h$ 是 horizon 嵌入（$h \in \{0,2,\ldots,16\}$）;一个 transformer 同时支持 9 个 horizon 的 target 预测,通过 $s/b/\alpha$ 让每个 horizon 在共享 backbone 上 specialize。

## 优点

1. **参数高效**: 只用一份 backbone 服务多个条件
2. **条件信号显式**: 不像 cross-attention 间接,而是直接调制 normalization
3. **训练稳定**: AdaLN-Zero 的零初始化保证收敛平稳

## 代表工作

- DiT (Peebles & Xie)
- CDiT (Conditional Diffusion Transformer)
- [[DexFuture]]: 用于 horizon-conditioned 回归

## 相关概念

- [[FiLM Modulation]]: 类似的 scale/shift 条件化机制
- [[Horizon-Conditioned Transformer]]
