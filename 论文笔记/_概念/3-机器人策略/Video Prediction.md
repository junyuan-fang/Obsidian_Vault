---
type: concept
aliases: [视频预测, Future Video Prediction, 未来视频生成]
---

# Video Prediction

## 定义
给定当前观测和条件信息（如动作、语言指令），预测未来视频帧序列的任务，是世界模型的核心能力之一。

## 核心要点
1. 可作为辅助监督信号提升策略学习（不一定在推理时使用）
2. 密集预测计算量大，稀疏预测（每隔 $\Delta$ 步）更高效
3. 评估指标包括 PSNR、SSIM 等图像质量指标

## 代表工作
- [[GigaWorld-Policy]]: 将视频预测作为训练时辅助监督，推理时可选
- [[Motus]]: 推理时必须生成视频
- [[Cosmos-Policy]]: 先生成视频再解码动作

## 相关概念
- [[WAM]]
- [[IDM]]
- [[Diffusion Transformer]]
