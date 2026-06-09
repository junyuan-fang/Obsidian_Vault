---
type: concept
aliases: [Horizon-Conditioned Target Transformer, Multi-Horizon Transformer]
---

# Horizon-Conditioned Transformer

## 定义

一种**用同一 transformer backbone 同时输出多个未来 horizon 预测**的网络设计,通过 horizon 嵌入 $c_h$ 与 [[AdaLN|AdaLN]] 调制,使每个 horizon $h \in \mathcal{H}$ 的输出在共享权重下 specialize。由 [[DexFuture]] 提出用于双手灵巧操作的 future-state 预测。

## 数学定义

输入 memory $M$（来自历史 visuomotor token）:

$$
\begin{aligned}
Y_h^0 &= \text{slot}(h) + \text{frame embed} \\
Y_h^{\ell+1} &= \text{TransformerBlock}_\ell(Y_h^\ell,\ M;\ c_h) \\
\hat{Z}_{t+h} &= W_{out} Y_h^L
\end{aligned}
$$

每个 block 内部使用 [[AdaLN|AdaLN]] 注入 $c_h$:

$$
Y_h \gets Y_h + \alpha_{\bullet}(c_h)\cdot\text{sublayer}(\mathrm{AdaLN}(Y_h, c_h))
$$

## 关键设计点

1. **horizon-specific learnable slot**: 每个 $h$ 一组 query slot,负责区分不同未来步
2. **AdaLN scale/shift/gate 全部条件化**: 让网络对"近未来" vs "远未来" specialize
3. **memory 共享**: 历史编码 $M$ 一次计算,所有 horizon 共用,避免 9 次 backbone 前向
4. **稀疏 horizon set $\mathcal{H}=\{0,2,4,\ldots,16\}$**: 9 个 sparse target,中间靠 linear interp 补足

## 与替代方案对比

| 方案 | 参数效率 | 推理速度 | 训练复杂度 |
|------|---------|---------|-----------|
| 每 horizon 独立 head | 中 | 中 | 高（多 head 训练） |
| 自回归 rollout | 高 | **慢** | 中 |
| **Horizon-conditioned** | **高** | **快** | 中 |

## 在 [[DexFuture]] 中的消融

Table 3 实证 $h \in \{8, 16, 24\}$ 中 $h=16$ 最优:
- $h=24$: 难度过大,所有指标退化
- $h=16$: 最佳 trade-off
- $h=8$: 简单任务略好但策略指导时长不够

## 代表工作

- [[DexFuture]]: 提出此设计

## 相关概念

- [[AdaLN]]
- [[Future-State Visuomotor Target Predictor]]
- [[Action Chunking]]
