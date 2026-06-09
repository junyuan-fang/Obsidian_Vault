---
type: concept
aliases: [Supervised Fine-Tuning, 监督微调]
---

# SFT

## 定义

在预训练模型基础上，用 (输入, 目标输出) 配对数据做 teacher-forced 监督学习的阶段；大模型后训练 pipeline 的第一步。

## 数学形式

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y^*)\sim \mathcal{D}} \left[ \frac{1}{L}\sum_{t=1}^{L} \log \pi_\theta(y_t^* \mid y_{<t}^*, x) \right]
$$

## 核心要点

1. **优点**：训练稳定，易于多任务统一
2. **缺点**：策略熵迅速降低，分布塌缩到专家轨迹附近
3. **典型 pipeline**：SFT → [[DAgger]] → [[DAPO]] / [[GRPO]]
4. **在 VLN 中**：单 SFT 模型遇到自身 rollout 误差易崩

## 代表工作

- [[GN0]]: SFT 作为四阶段训练的第一步
- 几乎所有 LLM 后训练

## 相关概念

- [[DAgger]]
- [[DAPO]]
- [[Covariate Shift]]
