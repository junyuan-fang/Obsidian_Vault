---
type: concept
aliases: [HFR, Hallucination-Free Ratio, 无幻觉比例]
---

# Hallucination-Free Ratio (HFR)

## 定义

衡量生成式世界模型 rollout 在出现"显著幻觉"（例如物体凭空消失 / 物理违反 / 手臂跳变）之前可信的归一化长度。由人工标注每条 rollout 上的首次幻觉帧位置，再除以 rollout 总长，跨标注员、跨样本平均得到。

## 数学形式

$$
HFR = \frac{1}{M}\sum_{m=1}^M \frac{1}{N}\sum_{i=1}^N \frac{t_h^{(i,m)}}{T_i}
$$

- $M$: 标注员数
- $N$: rollout 数
- $t_h^{(i,m)}$: 第 $m$ 个标注员在第 $i$ 条 rollout 上标注的首次明显幻觉帧
- $T_i$: 该 rollout 总长

## 核心要点

1. 越大越好：1 表示整段 rollout 都没有显著幻觉。
2. 与 $|\Delta SR|$ 互补：前者衡量"视觉可信长度"，后者衡量"成败统计偏差"。
3. 强依赖人工标注，难大规模化（可视为 PiL-World 体系的待优化点）。
4. 适合做世界模型评测的 model selection 指标。

## 代表工作

- [[PiL-World]]：首次提出，三任务平均把 HFR 从 [[Ctrl-World]] 的 41.5% 提升到 70.1%。

## 相关概念

- [[LPIPS]]
- [[World Model]]
- [[Policy-in-the-Loop]]
