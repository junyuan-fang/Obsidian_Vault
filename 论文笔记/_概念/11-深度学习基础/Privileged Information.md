---
type: concept
aliases: [Privileged Info, 特权信息]
---

# Privileged Information

## 定义
训练阶段可获取、部署阶段不可获取的额外信息（如真值 3D 几何、物体位姿、摩擦系数），通常用于 teacher-student 蒸馏中提升 teacher 的表现。

## 核心要点
1. **训练用**：teacher 输入 privileged；
2. **部署不用**：student 仅看 observable；
3. **典型用例**：locomotion 控制中训练给真高度图，部署仅靠 proprioception 估计；
4. **GRAIL 用法**：生成阶段把相机参数 / 物体几何 / 人物形态都当 privileged，简化重建。

## 代表工作
- Lee et al. 2020 (quadruped locomotion)
- [[GRAIL]]

## 相关概念
- [[Domain Randomization]]
- [[Egocentric Visual Policy]]
