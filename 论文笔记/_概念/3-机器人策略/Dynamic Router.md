---
type: concept
aliases: [动态路由器, Routing Network, Reasoning Router]
---

# Dynamic Router

## 定义

在 [[Adaptive Multi-Modal Reasoning|自适应多模态推理]] 中，**根据当前上下文输出二进制门控 token 来决定调用哪条推理路径**的轻量分类网络。[[AdaWAM]] 中记为 $\mathcal{R}_\psi$，输出 $\langle\text{TR}\rangle$ 与 $\langle\text{VR}\rangle$。

## 数学形式

输入上下文向量：

$$
C_t = [\,v_t \,\|\, e_l \,\|\, e_{c_t}\,]
$$

其中 $v_t$ 是当前视觉 latent $z_t$ 的池化嵌入，$e_l$ / $e_{c_t}$ 分别是全局指令和当前 subtask 的文本嵌入。路由器输出：

$$
(\hat{y}_{\text{TR}},\, \hat{y}_{\text{VR}}) = \mathcal{R}_\psi(C_t),\quad \langle k \rangle = \mathbb{1}[\hat{y}_k > 0.5]
$$

## 核心要点

1. **必须轻**: 路由器本身的延迟不能反噬 video diffusion 节省的延迟，[[AdaWAM]] 用轻量 MLP head
2. **必须看到多模态上下文**: 视觉 + 全局指令 + 当前 subtask 三者拼接
3. **监督**: 用 trajectory cue + VLM 校验生成的启发式标签 $\hat{y}_k$ 做 [[Binary Cross-Entropy|BCE]]
4. **粒度 = action chunk**: 比逐 token 更稳定，比整条 episode 更细
5. **4 种推理模式组合**: TR/VR 各两态 → action-only / +text / +video / +both

## 代表工作

- [[AdaWAM]]: 路由器 + [[Video-Action DiT]] 联合训练 50K 步

## 相关概念

- [[Adaptive Multi-Modal Reasoning]]
- [[Binary Cross-Entropy]]
- [[Text Reasoning Module]]
