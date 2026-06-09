---
type: concept
aliases: [Task Graph, Abstract Transition Graph]
---

# Image-Grounded Task Graph

## 定义

[[STRIPS-WM]] (Ajith & Chamzas 2026) 提出的中间抽象层。从一组观察 $(o_t, a_t, o_{t+1})$ 三元组中：

1. 用 [[JEPA]] 风格 student-teacher 自监督 + [[FSQ]] 量化，得到每张图像的离散 code $z = \alpha(o) \in \hat S$
2. 在所有观察转移上构图 $\hat G = (\hat S, \hat E)$，节点是 code，边是 $(z_t, a_t, z_{t+1})$

## 关键性质

- **无算子结构**: 只是观察到的转移图，没有 precondition / add / delete
- **可被 BFS 搜**: 但只能找观察过的状态对的路径，**不能组合泛化**
- **作为 Stage 2 的输入**: 后续的 [[STRIPS]] 算子学习从该图中归纳出谓词与算子

## 与本文关系

是 [[STRIPS-WM]] 三阶段管线的中间产物。论文 Figure 1 展示了它在三种规划空间（图像 / 抽象任务图 / STRIPS 空间）中的位置。

## 相关概念

- [[FSQ]]
- [[JEPA]]
- [[STRIPS-WM]]
- [[Latent Space Roadmap]]
- [[Visual Aliasing]]
