---
type: concept
aliases: [ControlNet, Control-Net]
---

# ControlNet

## 定义
在预训练扩散模型上旁挂一个可训练的 trainable copy，注入额外控制信号（边缘、深度、姿态等），不破坏原模型。

## 核心要点
1. 代表 Zhang et al. 2023
2. Zero-conv 初始化保证开始时是 identity
3. 缺点：参数翻倍、对位姿等几何信号噪声敏感

## 代表工作
- [[minWM]]: 对比对象：PRoPE 不用 ControlNet 旁挂即可注入相机条件

## 相关概念
- [[Diffusion Model]]
