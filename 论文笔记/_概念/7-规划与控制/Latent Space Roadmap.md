---
type: concept
aliases: [LSR, Latent Roadmap]
---

# Latent Space Roadmap (LSR)

## 定义

Lippi et al. (IEEE T-RO 2023) 提出的视觉规划方法。基本思路：

1. 用 VAE / contrastive 把图像 embed 到连续 latent 空间
2. 聚类得到 **roadmap nodes**（节点）
3. 从观察到的转移 $(o_t, a_t, o_{t+1})$ 中提取**有向边**
4. 测试时把当前/目标图编码到 latent → 找最近 node → 图搜索 → 输出动作序列

## 与 STRIPS-WM 的对比

| 维度 | LSR | [[STRIPS-WM]] |
|---|---|---|
| 状态表示 | 连续 embedding + 聚类 | [[FSQ]] 离散 + [[STRIPS]] 谓词 |
| 算子结构 | **无**（边只是观察过的 $(s, a, s')$） | 显式 precondition / add / delete |
| 组合泛化 | 不能（只能走观察过的状态对） | 能（谓词组合可推未见过状态对） |
| 长 horizon | 退化 | 100% 成功 |

## 与本文关系

[[STRIPS-WM]] 把 LSR 作为 baseline，证明缺乏算子结构是 LSR 长 horizon 退化的根本原因。

## 代表工作

- Lippi et al. 2023: "Latent Space Roadmap for Visual Action Planning", IEEE T-RO 39(1)

## 相关概念

- [[Visual Rearrangement Task]]
- [[STRIPS-WM]]
- [[Image-Grounded Task Graph]]
