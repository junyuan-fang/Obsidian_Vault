---
type: concept
aliases: [OXE, Open-X-Embodiment, Open-X, OpenX]
---

# Open X-Embodiment (OXE)

## 定义

Open X-Embodiment Dataset：DeepMind + 20+ 学术 / 工业实验室在 2023–2024 联合发布的跨形态机器人操作数据集，~2.4M 条轨迹、200+ embodiment、500+ skill，是 [[VLA]] 预训练的事实标准 corpus（OpenVLA / [[Pi05]] / RT-X 等均基于它）。

## 核心要点

1. **规模**: ~2.4M 真机轨迹（截至 2024 release），跨夹爪 / 双臂 / 移动底盘 / 灵巧手
2. **统一格式**: 用 RLDS（Reinforcement Learning DataSet）格式封装，统一动作空间为 7-DoF（或 14-DoF 双臂）
3. **基准用法**: "OXE-Constrained" 设定指 *只允许* OXE 做预训练，禁用其他大规模视频，是 VLA 论文公平比较的标准约束（见 [[LARA]] Table 1）
4. **局限**: 多 embodiment 平均下来单一形态数据稀疏；缺乏统一的 instruction 标注质量；夹爪类占大头

## 代表工作

- RT-X: OXE 原始 paper
- [[VLA|OpenVLA]]: 第一个开源 OXE 预训练 VLA
- [[Pi05]] / [[LARA]] / GR00T-N1.6: 后续 VLA 均把 OXE 作 baseline 预训练数据

## 相关概念

- [[VLA]]
- [[Robot Manipulation]]
