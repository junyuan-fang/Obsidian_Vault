---
type: concept
aliases: [动作跟踪奖励, tracking reward]
---

# Motion Tracking Reward

## 定义
PPO 训练 humanoid 模仿策略时常用的奖励形式：对每个分量（根位姿、各 body 位置 / 朝向、速度）用指数核惩罚仿真状态与参考状态的偏差。

## 数学形式
$$
R^{motion}_t = \sum_i w_i \exp\!\left(-\frac{\|\tilde x_{i,t} - x_{i,t}\|^2}{\sigma_i^2}\right)
$$

## 核心要点
1. **平滑可导**：指数核在大偏差时仍非零，避免梯度消失；
2. **分量加权**：根位姿权重通常远大于末端；
3. **配合正则**：常加能量、抖动、关节限位惩罚。

## 代表工作
- DeepMimic, [[SONIC]], [[GRAIL]]

## 相关概念
- [[PPO]]
- [[Grasp Reward]]
- [[Whole-Body Controller]]
