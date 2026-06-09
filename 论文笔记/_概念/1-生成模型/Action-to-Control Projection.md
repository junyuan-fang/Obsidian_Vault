---
type: concept
aliases: [Action-to-Control Projection, 动作—视觉控制投影, Visual Action Conditioning]
---

# Action-to-Control Projection

## 定义

把数值化的机器人动作（关节角 / 末端位姿）通过**正运动学 + 相机投影**映射成图像平面上的可视化控制信号（如 gripper marker），再作为视觉条件输入到视频扩散模型。相比直接把数值向量做 cross-attention，这种"动作画进画面"的方式给扩散模型提供了**像素对齐的几何先验**。

## 数学形式

给定动作序列 $A = \{a_k\}$：

$$
p_k^{\text{world}} = \mathrm{FK}(a_k), \quad p_k^{\text{img}} = \Pi(p_k^{\text{world}}; K, [R|t]) , \quad c_k = \text{DrawMarker}(p_k^{\text{img}}, s_k)
$$

其中 $\mathrm{FK}$ 是正运动学，$\Pi$ 是相机投影，$s_k$ 是 gripper 开闭状态编码为 marker 大小。

## 核心要点

1. **纯几何过程，无可学习参数**——免费先验。
2. marker 大小可编码连续状态（开闭程度）。
3. 通常画在 head-view（全局视角），让扩散模型获得空间 anchor。
4. 缺点：对 head-view 遮挡 / wrist-primary 任务效果有限。

## 代表工作

- [[PiL-World]]：首次系统化作为闭环 chunk-wise 世界模型的条件机制。

## 相关概念

- [[Inverse Kinematics]]
- [[ControlNet]]
- [[Action-Conditioned Video Diffusion]]
- [[Action Chunking]]
