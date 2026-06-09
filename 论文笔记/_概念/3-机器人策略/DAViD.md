---
type: concept
aliases: [Image-to-Video HOI]
---

# DAViD

## 定义
训练-free 图到视频 [[Human-Object Interaction|HOI]] 合成方法，依赖手工首帧选择 + 视频扩散模型生成交互。

## 核心要点
1. 不需训练，直接调用 VFM；
2. 重建几何 / 尺度全部留给后处理，欠约束；
3. 在 [[GRAIL]] benchmark 上虽然交互评分最高，但物理可执行 SR 仅 24%。

## 代表工作
- [[GRAIL]]: 提出资产先验思路改进 DAViD 的欠约束问题

## 相关概念
- [[Video Foundation Model]]
- [[CHOIS]]
- [[HOIDiff]]
