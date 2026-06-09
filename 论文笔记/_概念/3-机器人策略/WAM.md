---
type: concept
aliases: [World-Action Model, 世界-动作模型]
---

# WAM

## 定义
World-Action Model 结合视频世界模型和动作预测，通过生成未来视觉动态来约束动作的物理合理性。

## 核心要点
1. 从视频生成模型初始化，具备场景动态理解
2. 联合预测未来帧和动作序列
3. 视频生成提供额外监督信号

## 代表工作
- Motus: 代表性 WAM baseline
- [[GigaWorld-Policy]]: action-centered WAM，推理时可选跳过视频生成

## 相关概念
- [[VLA]]
- [[World Model]]