---
type: concept
aliases: [动作块, Action Chunk, 动作分块]
---

# Action Chunking

## 定义
一种动作预测范式，模型一次性预测未来多步动作序列 $a_{t:t+p-1}$，而非逐步预测单个动作，以提升时序一致性和执行效率。

## 数学形式

$$
a_{t:t+p-1} = (a_t, a_{t+1}, \ldots, a_{t+p-1}) \sim \pi_\theta(\cdot \mid o_t, s_t, l)
$$

## 核心要点
1. 减少策略调用频率，降低推理延迟
2. 提供时序平滑性，避免逐步预测导致的抖动
3. 块长度 $p$ 是关键超参数，需平衡反应速度与一致性

## 代表工作
- [[ACT]]: Action Chunking with Transformers，首次系统提出动作块概念
- [[GigaWorld-Policy]]: 使用流匹配预测动作块

## 相关概念
- [[VLA]]
- [[Diffusion Policy]]
