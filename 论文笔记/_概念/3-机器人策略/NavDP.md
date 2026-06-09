---
type: concept
aliases: [Navigation Diffusion Policy]
---

# NavDP

## 定义

Cai et al., 2025 提出的导航专用 [[Diffusion Policy]]：把视觉 + 目标条件输入扩散网络，去噪输出可执行底层动作序列。

## 核心要点

1. **架构**：vision encoder + goal conditioning + DDPM denoiser
2. **训练目标**：噪声预测 + critic 障碍感知回归 + 跨模态对齐辅助
3. **在 dual-system 中的角色**：低层 Action Expert，把上层 VLM 输出的 BEV 轨迹转成平滑、避障的动作
4. **GN-BAE 改动**：去掉 depth 输入，仅用 RGB + 目标条件，便于真机部署

## 代表工作

- Cai et al., 2025: NavDP 原版
- [[GN0]]: 把 NavDP 作为分 follow / planning 双专家
- InternNav: 同样使用 dual-system 思想

## 相关概念

- [[Diffusion Policy]]
- [[DDPM]]
- [[Vision-Language Navigation]]
