---
type: concept
aliases: [物理 AI, Physical Intelligence]
---

# Physical AI

## 定义

NVIDIA 等推动的概念：能感知、理解、预测并作用于真实物理世界的 AI 系统。覆盖 [[Embodied AI|具身智能]]、自动驾驶、工业机器人、数字孪生，强调"模型 + 物理仿真 + 真机部署"的闭环。

## 核心要点

1. **四要素**: 感知（视觉/触觉）、世界模型（预测）、策略（决策）、执行（控制）
2. **数字-物理对偶**: 用 [[World Model|世界模型]] / 仿真器在数字侧训练，迁移到物理侧执行
3. **数据需求大**: 真机数据 + 合成数据 + 通用视频先验混合训练
4. NVIDIA 把 Physical AI 与 Generative AI / Agentic AI 并列为三大 AI 浪潮

## 代表工作

- [[Cosmos]] / [[Cosmos3]]: Physical AI 的 omnimodal backbone
- [[OmniDreams]]: 自动驾驶 Physical AI 闭环
- [[Alpamayo]]: 自动驾驶 VLA
- Isaac / Omniverse: NVIDIA 的物理仿真栈

## 相关概念

- [[Embodied AI]]
- [[World Model]]
- [[VLA]]
