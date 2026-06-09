---
type: concept
aliases: [LIBERO Benchmark]
---

# LIBERO

## 定义

经典 [[VLA]] benchmark，含四个 suite——**Spatial / Object / Goal / Long**，分别测试空间泛化、物体泛化、目标泛化和长程任务能力，是 2023+ 几乎所有 VLA/imitation learning 论文的"必报"指标。

## 核心要点

1. **四 suite 分工**:
   - Spatial: 物体位置变化
   - Object: 物体形状/颜色变化
   - Goal: 目标条件变化
   - Long: 多步长程任务
2. **指标**: 每个 suite 平均成功率 + 四 suite Avg
3. **典型 SOTA**: 2026 顶层方法平均 97–99%
4. **天花板效应**: 经典 LIBERO 各 suite 已接近饱和，新方法的差距越来越小

## 代表工作

- [[WLA]]: 平均 98.6% / +TTS 98.9%
- [[AdaWAM]]: LIBERO-Long 99.1%（自适应多模态推理的关键证据）
- π₀.₅ / Motus / Fast-WAM 等几乎所有 VLA 均在此评测

## 相关概念

- [[RoboTwin]]
- [[VLA]]
