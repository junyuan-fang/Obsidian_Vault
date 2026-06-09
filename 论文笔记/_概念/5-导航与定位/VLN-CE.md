---
type: concept
aliases: [Vision-Language Navigation in Continuous Environments]
---

# VLN-CE

## 定义

把 R2R 类指令任务从离散全景图节点搬到 Habitat + Matterport3D 连续控制环境的 VLN 基准（Krantz et al., 2020）。

## 核心要点

1. **动作空间**：`{<FWD> 0.25m, <LEFT> 15°, <RIGHT> 15°, <STOP>}`
2. **输入**：连续 ego-RGB / depth / pano，无 oracle 图连接性
3. **指标**：NE / SR / OS / SPL / TL
4. **常用 split**：R2R Val-Unseen 是公认的泛化测试场

## 代表工作

- [[GN0]]: SOTA on R2R Val-Unseen (NE 3.50, SR 67.7, SPL 63.4)
- [[StreamVLN]] / [[DualVLN]] / [[CorrectNav]]: 之前的强 RGB-only baseline

## 相关概念

- [[Vision-Language Navigation]]
- [[VLN-PE]]
