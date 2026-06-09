---
type: concept
aliases: [Motus模型]
---

# Motus

## 定义
一种统一的世界-动作模型，利用预训练通用模型和共享运动信息，采用 Mixture-of-Transformer（MoT）架构联合预测动作和视频。

## 核心要点
1. 使用双向注意力，推理时必须生成未来视频才能获得动作
2. 推理延迟高（3231ms on A100）
3. 仿真中表现优异但受限于推理速度

## 代表工作
- [[GigaWorld-Policy]]: 对比基线，GigaWorld-Policy 比 Motus 快 9 倍且真实世界成功率更高

## 相关概念
- [[WAM]]
- [[VLA]]
- [[Diffusion Transformer]]
