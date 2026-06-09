---
type: concept
aliases: [Robot Memory Benchmark]
---

# RMBench

## 定义

针对**长程记忆依赖**操作任务设计的 benchmark，含 Battery Try / Blocks Ranking / Cover Blocks / Press Button 四任务，要求 agent 在多步骤中维持"哪些子目标已完成"的语义状态，是揭露纯反应式 [[VLA]] 短板的强测试集。

## 核心要点

1. **核心难度**: 任务步骤之间无法仅靠当前帧判断进度，必须依赖历史/语义状态
2. **代表性**: Press Button 任务要求模型记住"已按下哪几个按钮"——纯 VLA baseline 表现接近 0
3. **结果分布**:
   - π₀.₅ / X-VLA 平均 5–7%
   - Mem-0（带显式记忆） 28.5%
   - [[WLA]] 56.5%（接近翻倍）
4. **诊断价值**: 通过 ablation 可看出**语言子任务监督**对长程记忆至关重要

## 代表工作

- [[WLA]]: 主推动力，证明 *language loss* 是长程记忆的关键
- Mem-0: 显式记忆模块基线

## 相关概念

- [[VLA]]
- [[World-Action Model]]
- [[Embodied AI]]
