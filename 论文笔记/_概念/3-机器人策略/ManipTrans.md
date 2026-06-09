---
type: concept
aliases: [Manipulation Transformer, ManipTrans Policy]
---

# ManipTrans

## 定义

早期 **target-conditioned manipulation transformer** policy,以未来 demonstration target 作为强条件,用 transformer encoder 输出动作。在 [[DexFuture]] 评估中作为 GT-target 路线的第二个 oracle baseline。

## 核心特点

- 输入：当前 state + future demo target
- Backbone：标准 transformer encoder
- 输出：动作（高斯或确定性）
- **依赖 GT target**: 与 [[PhysGraph]] 同样属于 privileged 路线

## 性能对照（[[DexFuture]] Table 1 平均 SR）

| Method | 平均 SR |
|--------|--------:|
| ManipTrans (GT) | ~51% |
| [[PhysGraph]] (GT) | ~66.52% |
| [[DexFuture]] (Pred) | ~59.69% |

ManipTrans 一般弱于 PhysGraph 是因为缺乏 per-link 结构化 token,直接 flatten 输入,损失了多链结构 inductive bias。

## 相关概念

- [[PhysGraph]]
- [[Target-Conditioned Structured Dexterous Policy]]
- [[Action Chunking]]
