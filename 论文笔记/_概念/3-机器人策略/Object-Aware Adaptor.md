---
type: concept
aliases: [物体感知适配器, latent adaptor]
---

# Object-Aware Adaptor

## 定义
冻结 [[SONIC]] 等预训练 [[Whole-Body Controller|全身控制器]]，仅训练一个轻量适配器策略 $\pi_\phi$，根据物体状态 $o_t$ 输出 latent 残差 $\Delta z_t$ 注入 SONIC 潜空间，同时输出离散手指原语 $a^{hand}_t$。

## 数学形式
$$
(\Delta z_t, a^{hand}_t) = \pi_\phi(s_t, o_t),\quad a^{body}_t = G(z_t + \lambda \Delta z_t),\ \lambda = 0.1
$$

## 核心要点
1. **小 λ 缩放**：保留预训练全身先验，避免漂移；
2. **离散手指原语**：2-dim 二值映射到每手 7 DoF 抓握配置；
3. **观测设计**：物体位姿、手-物相对变换、接触力、[[BPS]] 形状、delta obs；
4. **训练**：[[PPO]]，奖励含 [[Motion Tracking Reward]] + 物体跟踪 + [[Grasp Reward]]。

## 代表工作
- [[GRAIL]]: 提出并验证在 humanoid loco-manipulation 上

## 相关概念
- [[SONIC]]
- [[FSQ]]
- [[Grasp Reward]]
