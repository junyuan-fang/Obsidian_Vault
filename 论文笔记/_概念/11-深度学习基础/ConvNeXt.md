---
type: concept
aliases: [ConvNeXt-Tiny, ConvNeXt-V2]
---

# ConvNeXt

## 定义

Liu et al. (CVPR 2022) 提出的纯 CNN backbone，借鉴 Transformer 的设计哲学（depthwise conv + LayerNorm + GELU + inverted bottleneck）改造 ResNet，在 ImageNet 上能匹敌 Swin Transformer。

## 变体

| 变体 | 参数量 | ImageNet top-1 |
|---|---|---|
| ConvNeXt-Tiny | 28M | 82.1% |
| ConvNeXt-Small | 50M | 83.1% |
| ConvNeXt-Base | 89M | 83.8% |
| ConvNeXt-Large | 198M | 84.3% |
| ConvNeXt-XL | 350M | 84.7% |

V2 (Woo et al. 2023) 加入 GRN + MAE 预训练，性能再提升。

## 在视觉规划中的用途

[[STRIPS-WM]] 用 **ConvNeXt-Tiny** 作为 Stage 3 视觉谓词分类器的 backbone：输入 $228 \times 228$ RGB，输出 $m$ 维 sigmoid 预测谓词。选 ConvNeXt 是因为它**比 ViT 在小数据集（3,000–12,000 样本）上更鲁棒**。

## 相关概念

- [[Self-Attention]]
- [[Binary Cross-Entropy]]
