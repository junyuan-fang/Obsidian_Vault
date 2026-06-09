---
type: concept
aliases: [轨迹分词, Action Tokenization]
---

# Trajectory Tokenization

## 定义

把连续动作或 waypoint 离散成 token 让自回归 LLM/VLM 通过 next-token prediction 输出轨迹的范式。

## 核心要点

1. **离散动作词表**：如 `<FWD> <LEFT> <RIGHT> <STOP>` 直接作为 LLM 输出符号
2. **连续 waypoint 离散化**：BEV 坐标归一化后映射到 $N$ 个 bin（如 GN-BAE 的 1000 桶 `<0>`～`<999>`）
3. **优点**：训练稳定 / 训练目标统一 / 多任务共享 vocabulary
4. **缺点**：discrete-to-continuous gap，需 RL 弥合（token 距离 ≠ 物理距离）

## 代表工作

- [[GN0]]: 离散动作 + 1000-bin waypoint 双轨
- RT-2 / OpenVLA: 离散动作 token
- Pix2Seq 系列: 早期 location token 思想

## 相关概念

- [[VLM]]
- [[SFT]]
- [[DAPO]]
