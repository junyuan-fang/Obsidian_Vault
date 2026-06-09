---
type: concept
aliases: [动作分块, Action Chunk, Chunked Prediction]
---

# Action Chunking

## 定义

策略一次预测未来 $H$ 步动作序列 $\mathbf{u}_{t:t+H}$ 而非单步,通过 chunk 复用降低 inference 频率、抑制误差累积，是 [[Diffusion Policy]] / ACT / [[VLA]] 类策略的标配技巧。

## 数学形式

$$
\mathbf{u}_{t:t+H} = \pi_\theta(o_{t-K+1:t}, l), \quad \text{执行 } \mathbf{u}_t, \dots, \mathbf{u}_{t+H'-1}\, (H' \le H)
$$

每个 chunk 通常重叠或部分执行后重规划。

## 核心要点

1. **降低决策频率**: 50Hz 控制只需 5-10Hz 策略推理
2. **抑制 covariate shift**: 一次预测多步比逐步预测稳定
3. **chunk 长度 vs 响应性的 trade-off**: 太长丢失反应能力，太短失去优势
4. **temporal ensembling**（ACT 提出）: 重叠 chunk 加权融合,平滑动作

## 代表工作

- ACT (Action Chunking Transformer)：奠基工作
- [[Diffusion Policy]]：扩散版 chunking
- [[Pi05]] / [[Alpamayo]] / [[Cosmos3]]：现代 VLA 均默认 chunk 输出

## 相关概念

- [[VLA]]
- [[Diffusion Policy]]
- [[Trajectory Tokenization]]
