---
type: concept
aliases: [视觉重排任务, Rearrangement Planning]
---

# Visual Rearrangement Task

## 定义

机器人视觉规划的标准评估场景：给定当前 RGB 图 $o_0$ 和目标 RGB 图 $o_g$，输出一串高层动作 $a_1, ..., a_T$ 让物体从当前布局重排到目标布局。常见域：

- **BlocksWorld**: 数个积木放在数个区域，动作 = `pick(block_i) → place(region_j)`
- **DinnerTable**: 几个餐具/物体放在桌面上的几个固定位
- **Habitat Rearrangement**: 室内场景中搬物体

## 评测指标

- **Planning success rate**: 执行返回动作序列后，环境状态是否到达目标
- **Plan length**: 比最短路径长多少
- **Horizon scaling**: 成功率随最短路 horizon 的退化曲线（[[STRIPS-WM]] Figure 3 的关键指标）

## 与本文关系

[[STRIPS-WM]] 在三个 visual rearrangement 域上评测：BlocksWorld、DinnerTable (sim)、DinnerTable Real。

## 代表工作

- [[STRIPS-WM]] (Ajith & Chamzas 2026)
- [[LatPlan]] (Asai et al. 2022)
- [[Latent Space Roadmap]] (Lippi et al. 2023)
- AI2-THOR Rearrangement Challenge (Weihs et al. 2021)

## 相关概念

- [[Propositional Planning]]
- [[STRIPS]]
