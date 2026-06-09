---
type: concept
aliases: [Seen-Action 设置, Seen-Action]
---

# Seen-Action Baseline

## 定义

跨实体/跨任务视频学习实验中的**对照设置**：仅用已见任务的 *带动作标签* 演示数据训练，然后直接在 *未见任务* 上评测，不引入任何视频先验。是衡量"视频监督带来多少增益"的核心 baseline。

## 核心要点

1. **作用**: 作为 *+Video* / *+Same-Emb. Video* / *+Cross-Emb. Video* 实验的下界
2. **对比维度**: 平均成功率差值直接反映 World Expert 从动作-free 视频中提取的迁移价值
3. **典型数字**: 在 [[WLA]] RoboTwin 5 个未见任务实验中，Seen-Action ≈ 13.0/11.6，加同实体视频后涨到 34.4/30.0

## 代表工作

- [[WLA]]: Table 3 用 Seen-Action 作为跨实体视频实验下界

## 相关概念

- [[VLA]]
- [[World-Action Model]]
- [[Embodied AI]]
