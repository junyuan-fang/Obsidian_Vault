---
type: concept
aliases: [GRAIL Joint Optimization Loss]
---

# GRAIL Reconstruction Loss

## 定义
[[GRAIL]] 在 4D HOI 联合重建阶段使用的复合损失，把 2D 关键点 / 2D 投影 / metric 深度 / 接触 / 时序正则统一到一个能量函数中。

## 数学形式
$$
\mathcal{L} = \lambda_{kp}\mathcal{L}_{kp} + \lambda_{proj}\mathcal{L}_{proj} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_{cont}\mathcal{L}_{cont} + \lambda_{reg}\mathcal{L}_{reg}
$$

## 核心要点
1. **多源约束**：图像证据 + 几何先验 + 接触语义；
2. **优化对象**：人体 / 物体残差参数 $\{\Delta\Theta^H_t, \Delta\Theta^O_t\}$；
3. **消融**：去掉任一项都会显著降低下游追踪 SR（Table 8）。

## 代表工作
- [[GRAIL]]

## 相关概念
- [[Keypoint Alignment Loss]]
- [[Depth Alignment Loss]]
- [[Chamfer Distance]]
