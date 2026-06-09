---
type: concept
aliases: [全身控制器, WBC]
---

# Whole-Body Controller (WBC)

## 定义
全身控制器统一管理 humanoid / quadruped 机器人所有关节自由度，使其在保持平衡（balance）和接触约束的同时执行高层运动指令。

## 核心要点
1. **数学形式**：通常解 QP 优化：
$$\min_{\ddot q, F} \|\ddot q - \ddot q^{des}\|^2 \quad \text{s.t.}\quad M\ddot q + h = S^T\tau + J^T F$$
2. **学习式 WBC**：[[SONIC]] / OmniH2O / ExBody 等用 PPO 端到端训练；
3. **关键挑战**：接触切换、力分配、动态稳定。

## 代表工作
- [[SONIC]]: GRAIL 使用的 latent 量化式 WBC
- [[GRAIL]]: 冻 SONIC + adaptor 的任务通用追踪

## 相关概念
- [[Loco-Manipulation]]
- [[PPO]]
- [[Isaac Lab]]
