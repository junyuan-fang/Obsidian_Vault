---
title: "GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors"
method_name: "GRAIL"
authors: [Tianyi Xie, Haotian Zhang, Jinhyung Park, Zi Wang, Bowen Wen, Jiefeng Li, Xueting Li, Qingwei Ben, Haoyang Weng, Yufei Ye, David Minor, Tingwu Wang, Chenfanfu Jiang, Sanja Fidler, Jan Kautz, Linxi Fan, Yuke Zhu, Zhengyi Luo, Umar Iqbal, Ye Yuan]
year: 2026
venue: arXiv
tags: [humanoid-robot, loco-manipulation, video-foundation-model, human-object-interaction, sim-to-real, motion-retargeting, whole-body-control]
zotero_collection: 5-具身智能与机器人/2-人形机器人
image_source: online
arxiv_html: https://arxiv.org/html/2606.05160v1
created: 2026-06-05
---

# 论文笔记：GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA + UCLA |
| 日期 | June 2026 |
| 项目主页 | https://research.nvidia.com/labs/dair/grail/ |
| 对比基线 | [[CHOIS]], [[HOIDiff]], [[DAViD]], [[HDMI]], [[ResMimic]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05160) / [HTML](https://arxiv.org/html/2606.05160v1) |

---

## 一句话总结

> GRAIL 用 3D 资产 + 视频生成模型 + 4D HOI 重建造出 2 万+ 条机器人可执行的 loco-manipulation 数据，全数字流水线训出真机部署的 Unitree G1 策略，抓取 84%、爬楼 90% 成功率。

---

## 核心贡献

1. **资产先验式 4D HOI 生成**：先指定 3D 物体、场景、相机、机器人匹配的人物形态，再用 [[Video Foundation Model|VFM]] 合成视频，最后重建 4D HOI，把"几何/尺度/形态推断"从重建阶段提前到生成阶段。
2. **20,000+ 仿真可执行序列**：覆盖 pick-up / whole-body manipulation / sitting / terrain traversal，1000 个物体 + 1000 个程序化地形，单序列摊销训练成本 0.5–0.9 分钟。
3. **任务通用追踪框架**：基于 [[SONIC]] 全身控制器，操控任务用 **object-aware latent adaptor**（冻 SONIC、训残差），地形/坐姿用 **scene-aware height encoder**（端到端微调）。
4. **零真机数据 Sim-to-Real**：egocentric 视觉策略仅靠 GRAIL 合成数据训练，Unitree G1 上抓取 seen 84% / unseen 80%，爬楼 90%。

---

## 问题背景

### 要解决的问题

人形 [[Loco-Manipulation|loco-manipulation]] 需要跨物体、跨场景、跨全身动作的大规模演示。[[Teleoperation|遥操作]] 和 [[Motion Capture|动捕]] 受限于物理布置、惯导服与机器人形态，难以规模化。

### 现有方法的局限

- **动捕数据**（GRAB, BEHAVE, ARCTIC）昂贵且少；
- **野外视频重建**（PHOSA, CHORE, [[DAViD]], ZeroHSI）把"几何 / 尺度 / 人体形态"全留给重建阶段去猜，欠约束；
- **训练式 HOI 生成**（[[CHOIS]], [[HOIDiff]]）受限于动捕训练集分布，难以泛化到任意物体；
- **训练 humanoid 控制**（HumanX, VideoMimic）受限于演示形态与机器人形态错配（[[Morphology Mismatch|morphology mismatch]]）。

### 本文的动机

把 [[Video Foundation Model|VFM]] 当作"交互先验"而非"重建对象"——在生成视频之前，就把物体几何、相机内外参、人物形态全部锁死成与机器人匹配的 3D 配置，VFM 仅用来合成动作；重建阶段所有几何/尺度/形态都是 privileged information，求解 HOI 变成"已知 3D 锚点下的轨迹精化"。

---

## 核心方法

### 模型架构

GRAIL 是一条四阶段流水线（生成 → 重建 → 追踪 → 部署）：

- **输入**：3D 物体资产 $\mathcal{O}$ + 程序化场景 + 机器人形态约束的人物模板
- **中间产物**：单视角交互视频 + 4D HOI 轨迹 $\{\Theta^H_t, \Theta^O_t\}$
- **控制后端**：[[SONIC]] 预训练 [[Whole-Body Controller|whole-body controller]]
- **输出**：可在 [[Isaac Lab]] 中执行、可蒸馏到真机的 [[Egocentric Visual Policy|egocentric visual policy]]
- **真机**：[[Unitree G1]] 人形机器人 + Luxonis OAK-D W 头戴相机

### 模块 1: Robot-Centric Human Video Generation（机器人中心化人视频生成）

**设计动机**: 让生成视频天然匹配机器人形态与场景几何，避免下游 [[Motion Retargeting|动作重定向]] 时的尺度漂移。

**具体实现**:
1. 用 [[Infinigen]] 程序化生成室内场景（地板 / 桌面），通过刚体仿真把物体落到无穿透静态位姿；
2. 用预拟合到 G1 形态的人物模板（高度、身材与 G1 一致）放入 Blender 渲染首帧，相机内参 $K$ 与外参 $E$ 已知；
3. 用 VLM（ChatGPT）从渲染图反推交互文本指令；
4. 用视频模型 [[Kling 2.5]] / [[Video Foundation Model|VFM]] 合成 5–10 秒、1920×1080、24 fps、静态相机的交互视频。

### 模块 2: Interaction-Aware HOI Reconstruction（交互感知 HOI 重建）

#### 2A 初始估计

- **人体动作**：用 [[GENMO]] 提取每帧 [[SMPL-X]] 位姿（body shape 锁死为模板形态），用 [[WiLoR]] 提取 [[MANO]] 手部参数，缺失帧线性插值 + [[Savitzky-Golay Filter|S-G 平滑]]；通过已知相机外参回到世界系，wrist [[Inverse Kinematics|IK]] 把 MANO 并入 SMPL-X。
- **物体追踪**：在合成数据上微调 [[FoundationPose]] 5 个 epoch（depth 通道置零），从已知首帧位姿向后传播，用 [[SAM2]] 分割掩膜校验一致性。

#### 2B 联合优化

在所有帧上联合优化残差 $\{\Delta\Theta^H_t, \Delta\Theta^O_t\}$，损失见下方[[GRAIL Reconstruction Loss|关键公式]]。

- **关键点对齐**：把 SMPL-X 投影到 2D 与检测关键点对齐；
- **物体投影对齐**：保持图像 2D 投影一致性；
- **深度对齐**：[[MoGe-2]] 估计深度，对齐已知背景深度恢复 [[Metric Depth|metric scale]]，再用 [[Chamfer Distance|Chamfer 距离]] 拉点云；
- **接触对齐**：VLM 给出每帧接触标签，仅在重叠投影区域罚 z 偏差；
- **正则**：脚滑（接触标签）+ 骨盆速度（[[GENMO]] 先验）+ 时序光滑。

### 模块 3: Task-General Loco-Manipulation Tracking（任务通用追踪）

#### 3A 动作重定向

用 [[GMR]] 把 SMPL-X 序列重定向到 G1 关节空间 $\tilde q_t$；因为模板已与 G1 形态对齐，[[Morphology Mismatch|形态错配]] 被显著压低，手-物与脚-地接触关系得以保留。

#### 3B Object-Aware Latent Adaptor（操控任务）

**设计动机**：操控任务里 [[SONIC]] 的全身先验已经够强，无需重训，只在 [[FSQ]] 量化前注入"物体感知"残差。

**具体实现**：冻结 SONIC 全部模块（encoder、quantizer、decoder），训练适配器：

- 观测 $s_t$ = 本体感知，$o_t$ = 物体位姿 / 手-物相对变换 / [[BPS|BPS 形状编码]] / 接触力 / delta obs；
- 输出 64-dim 残差 $\Delta z_t$ + 2-dim 二值手指原语 $a^{hand}_t$；
- 手指原语映射到每手 7 个 DoF 的预定义抓握；
- 残差按 $\lambda=0.1$ 缩放后注入 FSQ 前的潜变量。

#### 3C Scene-Aware Tracker（地形 / 坐姿）

**设计动机**：地形与坐椅需要"看到地"，需在 SONIC encoder 输入端补一张高度图。

**具体实现**：端到端微调 SONIC（含 encoder/quantizer/decoder），加入 11×11 局部高度图，经 2D-CNN（通道 [64, 128, 256]，3×3 步长 2）投影后拼接到 encoder；高度图由射线投射构造，转到机器人 yaw 对齐的局部系；辅助一个 kinematic decoder 监督动作重建。

#### 3D 奖励设计

[[PPO]] 训练，奖励由 [[Motion Tracking Reward|动作跟踪]] + [[Object Tracking Reward|物体跟踪]] + [[Grasp Reward|抓握]] + 正则项组成；操控用全套，地形/坐姿只用动作跟踪与正则。

### 模块 4: Sim-to-Real Deployment

把 latent 策略蒸馏成"看 RGB 出 SONIC token"的 [[Egocentric Visual Policy|egocentric visual policy]]，加 [[Domain Randomization|域随机]]，10 Hz 在 RTX 5090 上推理后通过有线连到 [[Unitree G1]]。

---

## 关键公式

### 公式 1: [[GRAIL Reconstruction Loss|总重建损失]]

$$
\mathcal{L} = \lambda_{kp}\mathcal{L}_{kp} + \lambda_{proj}\mathcal{L}_{proj} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_{cont}\mathcal{L}_{cont} + \lambda_{reg}\mathcal{L}_{reg}
$$

**含义**：4D HOI 联合重建的整体目标，把"2D 像素证据 + 3D 深度证据 + 接触约束 + 时序平滑"四类信号统一到一个能量上。

**符号说明**：
- $\mathcal{L}_{kp}$：关键点对齐损失（SMPL-X 关键点 vs 检测）
- $\mathcal{L}_{proj}$：物体 2D 投影一致性
- $\mathcal{L}_{depth}$：深度 / Chamfer 对齐（恢复 metric scale）
- $\mathcal{L}_{cont}$：接触帧的 z 方向贴合
- $\mathcal{L}_{reg}$：脚滑、骨盆速度、时序光滑等正则
- $\lambda_*$：各项权重系数

### 公式 2: [[Keypoint Alignment Loss|关键点对齐损失]]

$$
\mathcal{L}_{kp} = \frac{1}{T}\sum_{t=1}^{T} \left\| \mathcal{K}^H(\Theta^H_t) - p_t \right\|
$$

**含义**：把 SMPL-X 关键点投到 2D，与检测得到的 2D 关键点 $p_t$ 做 L1 距离，约束人体位姿不漂离图像证据。

**符号说明**：
- $\mathcal{K}^H(\Theta^H_t)$：参数 $\Theta^H_t$ 下的 SMPL-X 关键点 2D 投影
- $p_t$：图像检测得到的 2D 关键点
- $T$：序列总帧数

### 公式 3: [[Depth Alignment Loss|深度对齐损失]]

$$
\mathcal{L}_{depth} = \frac{1}{T}\sum_{t=1}^{T}\Big[\mathrm{CD}(V^{H,vis}_t, P^H_t) + \mathrm{CD}(V^{O,vis}_t, P^O_t)\Big]
$$

**含义**：用 [[MoGe-2]] 估计的 metric 深度反投得到的人/物点云 $P^{H/O}_t$ 与 SMPL-X / 物体网格可见顶点 $V^{H/O,vis}_t$ 之间求 [[Chamfer Distance|Chamfer 距离]]，把 3D 几何对齐到物理可信尺度。

**符号说明**：
- $V^{H,vis}_t, V^{O,vis}_t$：当前帧人体 / 物体网格的可见顶点
- $P^H_t, P^O_t$：从估计深度反投回 3D 的人 / 物点云
- $\mathrm{CD}$：Chamfer Distance

### 公式 4: [[Object-Aware Adaptor|Object-Aware Adaptor 策略]]

$$
(\Delta z_t,\ a^{hand}_t) = \pi_\phi(s_t, o_t),\quad
a^{body}_t = G(z_t + \lambda\, \Delta z_t),\quad \lambda = 0.1
$$

**含义**：在冻结的 [[SONIC]] 控制器之上，由策略 $\pi_\phi$ 同时输出 64 维潜空间残差 $\Delta z_t$ 与 2 维手指原语 $a^{hand}_t$；残差以小系数 $\lambda$ 注入 [[FSQ]] 前的潜变量并被解码器 $G$ 译为身体动作。

**符号说明**：
- $\pi_\phi$：可训练适配器策略（PPO 学习的唯一组件）
- $s_t$：本体感知状态
- $o_t$：物体位姿 / 接触 / [[BPS|BPS]] 形状 / delta obs
- $G$：SONIC 解码器（保持冻结）
- $\lambda$：残差缩放系数（=0.1）

### 公式 5: [[Motion Tracking Reward|动作跟踪奖励]]

$$
R^{motion}_t = \sum_i w_i \exp\!\left(-\frac{\|\tilde x_{i,t} - x_{i,t}\|^2}{\sigma_i^2}\right)
$$

**含义**：每个分量（根位姿、各 body 位置/朝向、速度等）以指数核惩罚仿真状态 $x_{i,t}$ 与参考 $\tilde x_{i,t}$ 的偏差，方便 PPO 平稳收敛。

**符号说明**：
- $\tilde x_{i,t}$：参考动作分量
- $x_{i,t}$：仿真观测分量
- $w_i, \sigma_i$：分量权重与带宽

### 公式 6: [[Grasp Reward|抓握奖励]]

$$
R^{grasp}_t = w_c \min\!\left(\frac{N^{contact}_t}{N_{min}}, 1\right) + w_d \big[-\cos(d^{thumb}_t, d^{index}_t)\big]_+ + w_f \exp\!\left(-\gamma \cdot \frac{1}{N_f}\sum_j \|f_{j,t}-c_t\|\right)
$$

**含义**：三段式抓握奖励——(1) 接触帧数饱和到 $N_{min}$；(2) 拇指/食指方向相向（cos<0）才计分，鼓励对捏；(3) 指尖到物体中心 $c_t$ 距离指数衰减。

**符号说明**：
- $N^{contact}_t$：当前接触手指数
- $d^{thumb}_t, d^{index}_t$：拇指 / 食指方向
- $[\cdot]_+$：截断到非负
- $f_{j,t}$：指尖 $j$ 的位置
- $c_t$：物体中心

---

## 关键图表

### Figure 1: 全数字数据生成 → 真机部署

![Figure 1](https://arxiv.org/html/2606.05160v1/figures/teaser_v8.jpeg)

**说明**: GRAIL 不依赖物理场景重建或遥操作即可生成 humanoid loco-manipulation 数据。上：仿真 humanoid 执行 pick-up、whole-body 操控、坐椅、地形通过等参考；下：仅用 GRAIL 数据训练的 egocentric 视觉策略部署到 [[Unitree G1]]，完成爬楼与抓取。

### Figure 2: Asset-Conditioned 4D HOI Generation

![Figure 2](https://arxiv.org/html/2606.05160v1/figures/hoigen.001.jpeg)

**说明**: 给定 3D 物体资产，先渲染含已知相机参数与机器人匹配人物的 3D 场景，再通过 VFM 合成静态相机交互视频，最后以关键点 / 深度 / 接触三类损失联合精化人 + 物轨迹，全过程锚定到已知 3D 配置（[[Privileged Information|privileged info]]）。

### Figure 3: 互补控制器适配的任务通用追踪

![Figure 3](https://arxiv.org/html/2606.05160v1/figures/motion_tracking.002.jpeg)

**说明**: 操控轨迹训"冻 SONIC + adaptor"路径，输出 latent 残差 + 手指原语；地形/坐姿轨迹端到端微调 SONIC 并补一个 scene-aware [[Height Map Encoder|高度图编码器]]。两条路径共享同一控制器骨架。

### Figure 4: 生成的 Loco-Manipulation 数据

![Figure 4](https://arxiv.org/html/2606.05160v1/figures/diverse_motion.005.jpeg)

**说明**: 仿真 Unitree G1 执行的代表性序列，覆盖 pick-up、whole-body 操控、坐椅、地形通过，跨越多样的物体与场景几何。

### Figure 5: Sim-to-Real 部署

![Figure 5](https://arxiv.org/html/2606.05160v1/figures/deployment.003.jpeg)

**说明**: 仅用 GRAIL 数据训练的视觉策略迁移到 Unitree G1，完成真实物体抓取与爬楼。

### Figure 6: HOI 生成定性对比

![Figure 6](https://arxiv.org/html/2606.05160v1/x1.png)

**说明**: 同样 20 物体测试集上，GRAIL 在接触准确、手姿自然方面优于 [[CHOIS]] / [[HOIDiff]] / [[DAViD]]，后者常出现"扁平死手"或静止手姿。

### Table 1: HOI 生成质量对比

| Method | Contact↓ | Pen.↓ | Inter. Score↑ | Human Smo.↓ | Obj Smo.↓ | SR↑ | Body Dev.↓ | Obj Dev.↓ |
|--------|----------|-------|---------------|-------------|-----------|------|------------|-----------|
| [[HOIDiff]] | 0.012 | 2.07% | 1.79 | 0.0043 | 0.0118 | 15.8% | 0.2120 | 0.3352 |
| [[CHOIS]] | 0.034 | 3.74% | 2.47 | 0.0055 | 0.0062 | 10.5% | 0.2564 | 0.3642 |
| [[DAViD]] | 0.246 | 1.46% | 2.74 | 0.0024 | 0.0605 | 24.0% | 0.4723 | 0.5826 |
| **GRAIL** | **0.008** | **0.90%** | **3.58** | **0.0033** | **0.0022** | **88.9%** | **0.0913** | **0.0851** |

**说明**: 物理可执行性（[[InterMimic]] 重放成功率）从最高 24% 飙升到 88.9%，对应 Body / Obj 偏差降为基线的 1/3–1/5。

### Table 2: 任务通用追踪 + 消融

| Method | SR↑ | ObjPos↓ | MPJPE-L↓ |
|--------|-----|---------|----------|
| [[HDMI]] | 48.5% | 0.283 | 122.3 |
| [[ResMimic]] | 49.2% | 0.393 | 80.9 |
| Ours w/o [[SONIC]] | 45.0% | 0.395 | 243.5 |
| Ours w/o $\pi_\phi$ | 39.7% | 0.303 | 37.1 |
| Ours w/o Rel. Obs. | 57.9% | 0.257 | 43.0 |
| **Ours (Full)** | **81.4%** | **0.135** | **41.8** |

**关键发现**: 去掉 SONIC 先验则身体追踪崩（243.5 MPJPE-L），去掉 adaptor $\pi_\phi$ 则 SR 跌到 39.7%——证明"身体复刻"与"物体适配"两条腿缺一不可。

### Table 3: 真机抓取（seen / unseen）

| Seen 物体 | 成功率 | Unseen 物体 | 成功率 |
|-----------|--------|--------------|--------|
| Cube | 100% | Spray Can | 100% |
| Apple | 60% | Lint Roller | 50% |
| Tea Box | 100% | Peach | 90% |
| Carrot | 70% | Flashlight | 80% |
| Wet Wipes | 90% | Medicine Bottle | 80% |
| **平均** | **84%** | **平均** | **80%** |

**说明**: 真机爬楼 90% 成功率（未列在表中），全部数据均零真机采集。

### Table 4: 生成时长（A100，5 秒序列）

| 阶段 | 时间 |
|------|------|
| Video 生成 (Kling API) | ~1 min |
| 人体动作估计 | ~2 min |
| 物体位姿追踪 | ~1 min |
| 优化预处理 | ~2 min |
| 联合优化 | ~8 min |
| **总计** | **~14 min** |

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Robocasa + ComAsset + [[OMOMO]] + Hunyuan3D | 1,000 物体 | 跨类别 3D 资产 | 操控生成 |
| 程序化地形 ([[Infinigen]]) | 1,000 配置 | 楼梯 / 斜坡 / 路缘 | 地形通过 |
| 生成 HOI 序列 | 20,000+ | 4 类任务 | 训练 trackers |

### 实现细节

- **VFM**: [[Kling 2.5]] Turbo Pro，1920×1080 @ 24 fps，5–10 s 静态相机
- **人体估计**: [[GENMO]]（body）+ [[WiLoR]]（hand），合并到 [[SMPL-X]]
- **物体追踪**: [[FoundationPose]]（合成数据 fine-tune 5 epoch）+ [[SAM2]] 校验
- **重定向**: [[GMR]] → [[Unitree G1]] 关节空间
- **控制器**: [[SONIC]] 全身控制器；[[FSQ]] 量化潜空间
- **仿真**: [[Isaac Lab]]；[[PPO]] 64×L40 GPU，每卡 1024 envs，30k iter
- **真机**: Unitree G1 + Luxonis OAK-D W + RTX 5090 桌面侧 10 Hz 推理

### 关键结果

1. **HOI 生成质量**：[[InterMimic]] 重放成功率 88.9%，是最强基线（[[DAViD]] 24%）的 3.7×；
2. **追踪策略**：标准 124 动作集上 SR 81.4%，比 [[HDMI]]/[[ResMimic]] 高 30+；
3. **真机部署**：抓取 seen 84%（5 物体）/ unseen 80%（5 物体），爬楼 90%；
4. **成本**：单序列摊销训练 0.5–0.9 min，数据生成 14 min/5 秒。

---

## 批判性思考

### 优点

1. **资产先验思路解耦了"生成"与"重建"**：把欠定问题（从单视频反推 4D HOI）变成"已知锚点下的精化"，是这条流水线能 work 的根本。
2. **任务通用性强**：同一框架覆盖 manipulation / locomotion / sitting，对工业部署的样本效率非常友好。
3. **互补控制器设计精妙**：操控冻 SONIC 只动 adaptor，地形端到端微调——两种适配在数据量与目标分布上其实非常不同，分开处理避免互相伤害。
4. **真机数据零依赖**：只用合成数据就能做到 84% / 80% / 90%，对没有动捕棚的实验室是巨大利好。

### 局限性

1. **依赖闭源 VFM (Kling)**：生成可复现性受 API 价格 / 政策影响；
2. **失败过滤丢弃比例未透露**：14 min/序列只是"过滤后"，真实生成成本更高；
3. **重建在遮挡 / 快动作下退化**：抓取小物体（Apple 60%、Lint Roller 50%）成功率明显偏低；
4. **任务族切换需重训**：adaptor / scene-aware tracker 都是任务族级别的，跨任务族没有零样本能力；
5. **G1 形态绑定**：换机器人形态需要重新生成模板与微调，并非真正"形态无关"。

### 潜在改进方向

1. 用开源 VFM（[[Cosmos]] / [[Wan2.2]]）替换 Kling，做封闭流水线；
2. 把 adaptor 与 scene-encoder 进一步统一成"条件化 [[VLA]]"，跨任务族复用；
3. 引入主动重生成机制：失败序列回馈 prompt，二次 VFM 调用替换；
4. 拓展到双手协同 / 移动操控（mobile manipulation）。

### 可复现性评估

- [x] 项目主页（research.nvidia.com/labs/dair/grail/）
- [ ] 代码开源（截至 v1 未见）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（PPO 设置、奖励权重、网络结构都给了）
- [ ] 数据集可获取（生成的 20K 序列未承诺公开）

---

## 相关概念

### 基于
- [[SONIC]]: 提供冻结的全身控制先验，是 adaptor / scene tracker 的共同骨架
- [[Video Foundation Model|VFM]]: 提供"动作合成"先验，是生成阶段的核心
- [[FoundationPose]]: 物体追踪的初始化
- [[GENMO]]: 人体动作初始化
- [[GMR]]: SMPL-X → Unitree G1 重定向

### 对比
- [[CHOIS]]: 训练式语言条件 HOI 生成
- [[HOIDiff]]: 扩散式 affordance HOI 生成
- [[DAViD]]: 训练-free 图到视频 HOI 生成
- [[HDMI]]: 人形机器人 manipulation 基线
- [[ResMimic]]: 残差模仿基线

### 方法相关
- [[Loco-Manipulation]]: 任务范式
- [[Human-Object Interaction|HOI]]: 数据形态
- [[Whole-Body Controller]]: 控制器范式
- [[Object-Aware Adaptor]]: 本文新提的 adaptor 模块
- [[Motion Tracking Reward]]: PPO 奖励设计
- [[Grasp Reward]]: 三段式抓握奖励
- [[GRAIL Reconstruction Loss]]: 4D HOI 联合重建损失

### 数据/技术栈
- [[SMPL-X]], [[MANO]]: 人体 / 手参数化
- [[Chamfer Distance]]: 点云对齐度量
- [[BPS]]: 物体形状编码
- [[FSQ]]: SONIC 潜空间量化
- [[Domain Randomization]]: Sim-to-Real 关键技巧
- [[Isaac Lab]]: 仿真训练平台
- [[Infinigen]]: 程序化场景生成
- [[Unitree G1]]: 真机平台

---

## 速查卡片

> [!summary] GRAIL
> - **核心**: 用 3D 资产锚定 + VFM 合成视频，反推 4D HOI 训人形 loco-manipulation 策略
> - **方法**: 资产先验生成 → 联合优化重建 → adaptor (操控) / scene encoder (地形) → 视觉策略蒸馏
> - **结果**: 20K+ 序列；HOI SR 88.9%；追踪 SR 81.4%；真机抓取 84% / 爬楼 90%（零真机数据）
> - **代码**: 暂未开源（仅 project page）

---

*笔记创建时间: 2026-06-05*
