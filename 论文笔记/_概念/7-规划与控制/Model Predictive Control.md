---
type: concept
aliases: [MPC, 模型预测控制]
---

# Model Predictive Control

## 定义

Richalet et al., 1978 起源的滚动时域优化控制：在每个时间步基于动力学模型预测有限时域 $H$ 内的状态轨迹，求解最优控制序列，仅执行第一步，下一步重复求解。

## 数学形式

$$
a_t^* = \arg\min_{a_{t:t+H}} \sum_{k=0}^{H} \gamma^k C(s_{t+k}, s^*_{t+k})
$$

其中 $C$ 为代价函数（跟踪误差 + 控制能量 + 约束惩罚），$\gamma$ 为折扣。

## 核心要点

1. **优点**：显式约束、可处理非线性
2. **缺点**：需要可靠模型 + 求解算力
3. **在导航中**：常作为 oracle 给 VLN 模型提供离散动作标签
4. **配合 A\***：A\* 生成几何路径，MPC 跟踪路径输出离散动作

## 代表工作

- Richalet et al., 1978: 起源
- [[GN0]]: 用 MPC 把 A\* 路径转成 `<FWD/LEFT/RIGHT/STOP>` 标签做 [[DAgger]]

## 相关概念

- [[DAgger]]
