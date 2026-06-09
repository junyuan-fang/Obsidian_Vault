---
type: concept
aliases: [SMPL Expressive]
---

# SMPL-X

## 定义
SMPL-X 是 SMPL 系列的扩展版本，统一身体 + 手 + 脸的参数化人体模型，参数包含 body pose、shape、hand pose（MANO 兼容）、face expression（FLAME 兼容）。

## 数学形式
$$
V = M(\beta, \theta, \psi)
$$
其中 $\beta$ 为 shape，$\theta$ 为 pose，$\psi$ 为 expression。

## 核心要点
1. 顶点数：~10,475；
2. 关节数：55（含手指 + 颈 + 头）；
3. 是 HOI / 动捕 / 动作生成的事实标准；
4. 通常配合 [[MANO]] 表达手部细节。

## 代表工作
- Pavlakos et al. 2019, "SMPL-X"
- [[GENMO]], [[WiLoR]], [[GRAIL]] 等

## 相关概念
- [[MANO]]
- [[Motion Capture]]
- [[Motion Retargeting]]
