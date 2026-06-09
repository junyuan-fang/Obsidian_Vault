---
type: concept
aliases: [视觉运动历史, Visuomotor Observation, RGB + Proprioception History]
---

# Visuomotor History

## 定义

策略输入中的**视觉 + 本体感知历史**:

$$
\mathcal{O}_{t-K:t} = \{I_\tau,\ p_\tau\}_{\tau=t-K}^{t}
$$

- $I_\tau$: egocentric RGB（或 RGB-D）
- $p_\tau$: 本体感知 + 几何 cue（hand joint angle、wrist pose、tool/object anchor 等）
- $K$: 历史窗口长度

## 为何要用历史而不仅仅是当前帧

1. **运动估计**: 速度/加速度需要时间差分
2. **接触状态**: 接触/脱离是瞬时事件,单帧难以判断
3. **意图推断**: 长时程任务（如先抓再切）需要历史上下文
4. **遮挡鲁棒性**: 历史帧弥补当前帧的视觉遮挡

## 常见编码方式

| 方式 | 描述 | 代表 |
|------|------|------|
| Frame-wise vision encoder + temporal transformer | 每帧独立编码后时序聚合 | [[DexFuture]] |
| 3D CNN / Video transformer | 时空联合编码 | MVP / R3M |
| RNN/LSTM 流式 | 累积隐状态 | RT-1 早期 |
| Memory-augmented | 长期记忆 | RT-2 / RT-X |

## 在 [[DexFuture]] 中的角色

作为高层 [[Future-State Visuomotor Target Predictor]] 的唯一输入,完全替代 demo 特权 target —— 这是该论文的核心 contribution。

## 相关概念

- [[Future-State Visuomotor Target Predictor]]
- [[Action Chunking]]
- [[Egocentric Visual Policy]]
