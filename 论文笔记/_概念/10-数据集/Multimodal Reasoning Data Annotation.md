---
type: concept
aliases: [多模态推理数据标注, AdaWAM Annotation Pipeline]
---

# Multimodal Reasoning Data Annotation

## 定义

为 [[Dynamic Router|动态路由器]] 自动产生"该不该做文本/视觉推理"标签 $\hat{y}_k$ 的 pipeline，**不依赖人类逐 chunk 标注**——由 trajectory 启发式定位候选窗口 + [[VLM]] 语义校验组成。[[AdaWAM]] 首次系统化。

## 核心要点

1. **Trajectory-guided Subtask Annotation**: 从机器人轨迹提取 end-effector 位移、姿态变化、夹爪开闭、局部运动方差等特征 → 定位 subtask 候选窗口 → 用 Qwen3-VL-8B 做语义校验 → 输出 `<TR>` 标签；强制 subtask 序列单调递增
2. **Motion-Based Fine Manipulation Labeling**: 把机器人状态序列映射到 4 类运动模式
   - grasping
   - releasing
   - in-hand adjustment
   - local pose refinement

   后两类标为 `<VR>=1`
3. **关键优点**: 完全自动，可 scale 到任意长度的 demonstration 集
4. **潜在弱点**: 启发式可能 miss 微妙时刻，论文 Limitations 承认未来需要 RL fine-tune 路由器

## 代表工作

- [[AdaWAM]]: Section 3.1，详细描述两阶段标注流程

## 相关概念

- [[Dynamic Router]]
- [[Adaptive Multi-Modal Reasoning]]
- [[VLM]]
