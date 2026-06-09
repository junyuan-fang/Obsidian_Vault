---
type: concept
aliases: [LoRA, Low-Rank Adaptation]
---

# LoRA (Low-Rank Adaptation)

## 定义

通过给原模型权重矩阵 $W \in \mathbb{R}^{d\times k}$ 附加低秩增量 $\Delta W = BA$（$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}, r \ll \min(d,k)$）来实现参数高效微调的方法，原权重冻结，只训练 $A, B$。可显著降低显存占用并便于多任务切换。

## 数学形式

$$
h = W x + \Delta W x = W x + B A x, \quad r \ll \min(d, k)
$$

训练时常配合缩放因子 $\alpha / r$ 控制更新幅度。

## 核心要点

1. 显存友好：只需保存低秩部分梯度。
2. 推理时 $\Delta W$ 可吸收回 $W$，零额外延迟。
3. 多 LoRA 可叠加 / 切换，适合"一基座多任务"。
4. 视频扩散基座（Wan / SVD / Cosmos）微调几乎默认走 LoRA。

## 代表工作

- [[PiL-World]]：用 LoRA 在 [[Wan2.1]]-14B 上做任务微调，预训练 22 epoch / 微调 20 epoch。
- 大量 VLA 工作：在 VLM 基座上 LoRA 注入动作 head。

## 相关概念

- [[MoE]]
- [[Mixture-of-Transformers]]
