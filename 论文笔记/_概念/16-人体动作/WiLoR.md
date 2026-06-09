---
type: concept
aliases: [Hand pose estimator]
---

# WiLoR

## 定义
单帧 RGB 手部位姿估计模型，输出 [[MANO]] 参数。

## 核心要点
1. 输入：RGB 帧；
2. 输出：每只手的 MANO pose；
3. 缺失检测帧需后处理插值（如 [[Savitzky-Golay Filter]]）。

## 代表工作
- Potamias et al. 2025
- [[GRAIL]]

## 相关概念
- [[MANO]]
- [[SMPL-X]]
- [[GENMO]]
