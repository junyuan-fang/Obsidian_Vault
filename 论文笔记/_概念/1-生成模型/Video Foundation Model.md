---
type: concept
aliases: [VFM, 视频基础模型, 视频生成模型]
---

# Video Foundation Model (VFM)

## 定义
在大规模视频数据上预训练的扩散 / autoregressive 生成模型，可作为通用视频先验用于合成新视频或为下游任务提供动作 / 物理感知。

## 核心要点
1. **代表**：[[Cosmos]], [[Kling 2.5]], Wan, Sora, Veo；
2. **条件化**：文本 / 首帧 / 相机轨迹 / 动作；
3. **用作 prior**：可生成训练数据（[[GRAIL]]）、做世界模型（[[GAIA-1]]、[[Vista]]）。

## 代表工作
- [[GRAIL]]: 把 VFM 当作交互合成器
- [[Cosmos]], [[Kling 2.5]]

## 相关概念
- [[Video Diffusion Model]]
- [[Video World Model]]
- [[Diffusion Model]]
