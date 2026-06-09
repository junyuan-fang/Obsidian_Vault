---
type: concept
aliases: [抓握奖励]
---

# Grasp Reward

## 定义
针对灵巧手抓握任务设计的复合奖励，通常包含接触帧数、抓握姿态（拇指-食指对捏）和指尖到物体中心距离三个组成部分。

## 数学形式（[[GRAIL]] 中的实现）
$$
R^{grasp}_t = w_c\min\!\left(\frac{N^{contact}_t}{N_{min}}, 1\right) + w_d\big[-\cos(d^{thumb}_t, d^{index}_t)\big]_+ + w_f\exp\!\left(-\gamma \cdot \frac{1}{N_f}\sum_j \|f_{j,t}-c_t\|\right)
$$

## 核心要点
1. **接触饱和项**：避免奖励无界；
2. **方向项**：拇指 / 食指夹角 cos<0 才计分，强制对捏；
3. **距离项**：指数衰减，鼓励指尖接近物体中心。

## 代表工作
- [[GRAIL]]
- UniDexGrasp, DexCap

## 相关概念
- [[Motion Tracking Reward]]
- [[Object-Aware Adaptor]]
