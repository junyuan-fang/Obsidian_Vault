---
type: concept
aliases: [遥操作]
---

# Teleoperation

## 定义
人通过外部输入设备（动捕服 / VR / 主从机械手）远程操控机器人完成任务，用于采集模仿学习演示数据。

## 核心要点
1. **代价高**：需机器人 + 操作员 + 物理环境同时在场；
2. **数据质量好**：自然连贯，是 Diffusion Policy / ACT 等的主要数据来源；
3. **规模化难**：每条数据都要物理执行；
4. **替代方案**：[[Motion Capture|动捕]] 重定向、[[GRAIL]] 这种合成。

## 代表工作
- ALOHA, GELLO
- [[GRAIL]]: 主要动机就是规避遥操作的规模化瓶颈

## 相关概念
- [[Motion Capture]]
- [[Loco-Manipulation]]
