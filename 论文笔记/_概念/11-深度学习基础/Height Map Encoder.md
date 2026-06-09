---
type: concept
aliases: [高度图编码器, terrain encoder]
---

# Height Map Encoder

## 定义
将机器人脚下 / 周围的局部高度图（通常 5×5 ~ 21×21 网格）通过 2D CNN 编码为低维特征，再喂给控制策略，使机器人感知地形几何。

## 核心要点
1. **输入**：以机器人 yaw 对齐的局部 height map；
2. **架构**：2D CNN（通常 3 层）或 MLP；
3. **GRAIL 用法**：11×11 height map，CNN 通道 [64,128,256]，kernel 3×3 步长 2。

## 代表工作
- Lee et al. 2020
- [[GRAIL]]: scene-aware tracker 用 height encoder 处理地形

## 相关概念
- [[Whole-Body Controller]]
- [[Privileged Information]]
