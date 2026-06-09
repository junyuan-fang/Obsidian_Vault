---
type: concept
aliases: [Cosmos Policy]
---

# Cosmos-Policy

## 定义
一种两阶段世界模型方法，先利用大规模视频生成模型预测未来视频，再通过逆动力学模型从视频中解码动作。

## 核心要点
1. 利用 NVIDIA Cosmos 视频生成基础模型
2. 推理时需完整视频生成，延迟较高（~1413ms）
3. 真实世界成功率受限于视频生成质量和误差累积

## 代表工作
- [[GigaWorld-Policy]]: 作为对比基线，GigaWorld-Policy 在速度和成功率上均优于 Cosmos-Policy

## 相关概念
- [[WAM]]
- [[Video Prediction]]
- [[IDM]]
