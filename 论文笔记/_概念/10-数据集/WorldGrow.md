---
type: concept
aliases: []
---

# WorldGrow

## 定义

Li et al., 2025 提出的文本条件 3D 场景扩展方法：在小规模 3D 资产基础上用文本描述自动生成更大、更多样的室内场景。

## 核心要点

1. **流程**：3D 资产 → 结构化布局生成 → 物体填充 → 风格化（如"Victorian Vintage style"）
2. **作用**：缓解 3D 场景稀缺，提升数据多样性
3. **典型规模**：可生成 6×6 / 8×8 大小的复杂场景
4. **质量改进**：通常需对生成物体做表面细化

## 代表工作

- [[GN0]]: 在 GN-Matrix 中用 WorldGrow 扩展出 2000 个新场景

## 相关概念

- [[InteriorGS]]
- [[3D Gaussian Splatting]]
