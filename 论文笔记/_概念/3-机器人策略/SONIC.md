---
type: concept
aliases: [SONIC controller]
---

# SONIC

## 定义
SONIC 是基于离散潜空间（[[FSQ]] 量化）的人形机器人 [[Whole-Body Controller|全身控制器]]，将参考动作编码为 latent token 序列，由 decoder 解码为关节动作，可作为通用控制先验被下游适配。

## 核心要点
1. **三件套**：encoder $E$ + FSQ 量化 + decoder $G$；
2. **潜空间**：64 维 + FSQ 离散化，便于残差注入；
3. **任务适配**：可冻结后接 adaptor（如 [[GRAIL]] 的 object-aware adaptor），或端到端微调；
4. **训练**：大规模动捕参考 + [[PPO]] in [[Isaac Lab]]。

## 代表工作
- Luo et al. 2025, "SONIC"
- [[GRAIL]]: 把 SONIC 作为操控/地形双任务的共同控制骨架

## 相关概念
- [[Whole-Body Controller]]
- [[FSQ]]
- [[Object-Aware Adaptor]]
