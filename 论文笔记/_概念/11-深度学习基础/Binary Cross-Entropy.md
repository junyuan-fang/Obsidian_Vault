---
type: concept
aliases: [BCE, 二分类交叉熵, BCE Loss]
---

# Binary Cross-Entropy

## 定义

二分类任务的标准 loss：对真实标签 $y \in \{0,1\}$ 与预测概率 $\hat{y} \in (0,1)$，定义为：

$$
\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\big[\, y \log \hat{y} + (1 - y) \log (1 - \hat{y})\,\big]
$$

## 核心要点

1. **几何含义**: 等价于在伯努利分布 $\text{Bern}(\hat{y})$ 下最大化真实标签的对数似然
2. **梯度形式**: $\partial \mathcal{L}/\partial \hat{y} = (\hat{y} - y) / [\hat{y}(1-\hat{y})]$，配合 sigmoid 输出时数值稳定
3. **多标签场景**: 对多个独立二分类标签求和，如 [[AdaWAM]] 路由器同时输出 `<TR>` 和 `<VR>` 时把两路 BCE 相加
4. **类不均衡**: 常配 `pos_weight` 或 focal 项缓解正负样本悬殊
5. **与 Categorical CE 的关系**: 多类 softmax CE 在 K=2 时退化为 BCE

## 典型用途

- 二分类（疾病/非疾病、点击/未点击）
- 多标签分类（一张图多个属性独立 0/1）
- 路由/门控监督（如 [[Dynamic Router]]）
- 判别器训练（GAN）

## 相关概念

- [[Dynamic Router]]
- [[Adaptive Multi-Modal Reasoning]]
