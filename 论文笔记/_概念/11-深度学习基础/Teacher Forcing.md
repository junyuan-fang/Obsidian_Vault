---
type: concept
aliases: [Teacher Forcing, 教师强制]
---

# Teacher Forcing

## 定义
训练自回归模型时用 ground-truth 历史替代模型自己的生成作为下一步输入，加速收敛但引入 exposure bias。

## 数学形式
$$
$\hat y_t = f_\theta(y_{1:t-1}^{gt})$
$$

## 核心要点
1. 训练稳定、收敛快
2. 缺点：推理时模型必须用自己历史，分布偏移
3. 缓解：scheduled sampling、self-rollout

## 代表工作
- [[minWM]]: Stage 1 训练采用，Stage 3 用 self-rollout 弥补

## 相关概念
- [[Exposure Bias]]
- [[Autoregressive Diffusion]]
