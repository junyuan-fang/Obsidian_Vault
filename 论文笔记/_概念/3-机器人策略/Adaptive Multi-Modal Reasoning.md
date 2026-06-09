---
type: concept
aliases: [自适应多模态推理, AdaMM-Reasoning, Dreaming when Necessary]
---

# Adaptive Multi-Modal Reasoning

## 定义

在 [[World-Action Model|WAM]] / [[VLA]] 推理过程中，**按 action chunk 动态决定是否触发"文本推理"或"视觉推理"**，而不是每步都做全量预测的范式。由 [[AdaWAM]] 首次正式提出，由轻量 [[Dynamic Router|动态路由器]] 输出二进制 token 控制。

## 数学形式

记 chunk $t$ 的路由决策为 $(\langle\text{TR}\rangle_t, \langle\text{VR}\rangle_t) \in \{0,1\}^2$，则推理时多模态条件集 $\mathcal{S}_t$ 按下式自适应组装：

$$
\mathcal{S}_t = \{z_{\le t}\} \cup \{\tilde{c}_t \text{ if } \langle\text{TR}\rangle=1\} \cup \{\tilde{z}_f \text{ if } \langle\text{VR}\rangle=1\}
$$

## 核心要点

1. **动机**: 真正需要"做梦"的步骤通常只占长程任务的 10–30%，传统 WAM 在所有 chunk 都做视频预测属于算力浪费
2. **双路径职责正交**: 文本推理处理 subtask 切换（高层语义），视觉推理处理精细对准（低层物理）
3. **二进制门控 vs 软门控**: 离散开关比软权重更易解释、训练更稳
4. **延迟-精度帕累托**: 通过稀疏调度，能拿到 video-action joint 的精度 + action-only 的延迟
5. **训练**: 启发式自动标注 $\hat{y}_k$ + [[Binary Cross-Entropy|BCE]] 监督

## 代表工作

- [[AdaWAM]]: 首次系统化提出，在 [[LIBERO]]-Long 99.1% / [[RoboTwin]] 2.0 Hard 88.43% 上验证

## 相关概念

- [[Dynamic Router]]
- [[World-Action Model]]
- [[Text Reasoning Module]]
- [[Video-Action DiT]]
- [[Action Chunking]]
