---
type: concept
aliases: [OakInk-2, OakInk2 Dataset]
---

# OakInk2

## 定义

**双手 hands-object 接触轨迹数据集**,聚焦"复杂任务完成"中的 bimanual 工具使用 / 物体操控,包含 hand pose / object pose / contact 等多模态轨迹。是 [[DexFuture]] / [[PhysGraph]] / [[ManipTrans]] 等双手灵巧操作研究的主要 benchmark。

## 任务类别

| 类别 | 工具 | 对象 | 示例 task ID |
|------|------|------|--------------|
| Cut | chop knife | bread | 083f7@0 |
| Cut | fruit knife | apple | 9fc3e@0 |
| Pour | mug | mug | 1292e@0 |
| Wipe | big brush | whiteboard | 817fb@0 |
| Wipe | small brush | whiteboard | fc88d@0 |
| Shear | scissors | paper | e1fa6@0 / 9bb17@5 |
| Stir | spoon | container | 598a5@0 / cde36@1 |

## 评估指标

| 指标 | 含义 |
|------|------|
| **SR (%)** | 成功率（双手满足任务条件） |
| $E_t$ (cm) | tool/object translation error |
| $E_j$ (cm) | hand joint error |
| $E_{xt}$ (cm) | fingertip error |
| 3D / UV / PCK@5 / PCK@10 | future-state target prediction metric |

## 特点

1. **长时程**: 单任务通常 5-10s
2. **接触富集**: hand-tool-object 三者持续接触
3. **多模态 GT**: hand pose / object pose / contact label
4. **难度分布**: shear 类（窄接触区 + 突变）显著难于 cut/stir 类

## 代表工作

- [[DexFuture]]: 主评估 benchmark
- [[PhysGraph]]: target-conditioned baseline
- [[ManipTrans]]: 早期 baseline

## 相关概念

- [[Bimanual Dexterous Manipulation]]
- [[Human-Object Interaction]]
