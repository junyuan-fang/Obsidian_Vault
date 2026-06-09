---
type: concept
aliases: [Bird's Eye View Memory, 鸟瞰图记忆]
---

# BEV Memory

## 定义

把场景从俯视角（Bird's Eye View）正交投影成二维特征图，作为导航策略的紧凑空间记忆机制；保留 metric scale，规避透视前缩。

## 核心要点

1. **几何一致**：正交投影直接对应世界坐标，规划友好
2. **紧凑高效**：相对全景历史帧极大压缩 token 数
3. **VLM 友好**：作为单张图喂入 VLM，可被 visual encoder 直接处理
4. **来源**：可由 3DGS 顶视渲染、点云投影、或 BEVFormer 类预测得到

## 代表工作

- [[GN0]]: 首次形式化 3DGS 渲染的 BEV 作为 VLM 的"空间记忆"
- BEVFormer / LSS: 自动驾驶 BEV 的鼻祖

## 相关概念

- [[3D Gaussian Splatting]]
- [[Vision-Language Navigation]]
- [[VLM]]
