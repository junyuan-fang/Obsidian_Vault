---
type: concept
aliases: [文本推理模块, Subtask Predictor]
---

# Text Reasoning Module

## 定义

在 [[Adaptive Multi-Modal Reasoning|自适应多模态推理]] WAM 中**专门负责高层任务规划的 [[VLM]] 模块**——自回归预测下一 subtask 描述，仅在 [[Dynamic Router|路由器]] 输出 $\langle\text{TR}\rangle=1$ 时被调用。

## 数学形式

$$
\tilde{c}_t = \begin{cases}
\mathcal{V}_\omega(z_{\le t}, l) & \langle\text{TR}\rangle = 1 \\
c_t & \langle\text{TR}\rangle = 0
\end{cases}
$$

## 核心要点

1. **职责**: 任务转换点的高层语义规划（"我已经把杯子放好，下一步该擦桌子"）
2. **架构**: 通常用紧凑 [[VLM]]，[[AdaWAM]] 用 Qwen3-VL-4B
3. **训练**: Stage 2 单独微调（10K 步, lr 1×10⁻⁵），主干和路由器冻结，目标为 subtask 切换点的 NLL
4. **触发频率**: 长程任务上 [[AdaWAM]] 报告 `<TR>` 触发率在 15–30% 区间
5. **关键作用**: 是组合泛化的核心——消融后未见 subtask 组合成功率从 61% 塌缩到 3%

## 代表工作

- [[AdaWAM]]: 与 [[Video-Action DiT]] + [[Dynamic Router]] 协同的具体实现
- 也可类比 ACoT-VLA / CoT-VLA 等链式思维 VLA 的 textual rationale 路径

## 相关概念

- [[VLM]]
- [[Dynamic Router]]
- [[Adaptive Multi-Modal Reasoning]]
