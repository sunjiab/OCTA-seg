# OCTA 融合 OCT 图像的多模态血管分割研究路线

## 1. 课题目标

目标是利用配准的 **结构 OCT + OCTA** 信息，提高 OCTA 视网膜血管二值分割效果。

重点关注：

- 细小毛细血管
- 低对比度血管区域
- 血管断裂与连续性
- 噪声与投影伪影干扰

核心问题：

> OCT 这个额外模态，能否稳定提升 OCTA 血管分割效果？

建议前期先聚焦 **binary vessel segmentation**，暂时不做 artery / vein / FAZ 多任务。

---

## 2. 推荐起点：OCTA-500

优先从 **OCTA-500** 数据集开始，因为它天然适合 OCT + OCTA 多模态研究。

可重点关注的数据包括：

- OCT volume
- OCTA volume
- OCTA en-face projection
- Vessel ground truth
- FAZ / artery / vein 等其他标注

前期最重要的是先完全弄清楚：

```text
OCT volume shape
OCTA volume shape
OCTA projection shape
Vessel GT shape
各维度对应的空间方向
OCT / OCTA / projection / GT 之间的空间对应关系
```

---

## 3. 第一阶段：建立 OCTA 单模态 baseline

先不要设计复杂融合模型。

第一步先建立：

```text
OCTA projection
      ↓
    U-Net
      ↓
Vessel Mask
```

任务定义：

```text
background = 0
vessel     = 1
```

推荐 baseline：

- U-Net
- UNet++

输入：

```text
[B, 1, H, W]
```

输出：

```text
[B, 1, H, W]
```

### Loss

第一版可以直接使用：

```text
BCE Loss + Dice Loss
```

即：

\[
L = L_{BCE} + L_{Dice}
\]

### 评估指标

建议至少记录：

- Dice
- IoU
- Precision
- Recall / Sensitivity
- Specificity

后续可以增加：

- clDice

其中 clDice 对血管这类细长、拓扑结构明显的目标尤其有意义。

### 这一阶段的目标

不是创新，而是确保整个训练流程完全跑通：

```text
dataset
↓
dataloader
↓
augmentation
↓
model
↓
loss
↓
training
↓
validation
↓
evaluation
↓
visualization
```

---

## 4. 第二阶段：建立最简单的 OCT + OCTA 多模态 baseline

在 OCTA-only baseline 跑通后，再加入 OCT。

第一版不要上复杂 fusion block，先测试最简单的 early fusion：

```text
OCTA ──┐
       ├── concat ── U-Net ── segmentation
OCT  ──┘
```

假设：

```text
OCTA: [B, 1, H, W]
OCT : [B, 1, H, W]
```

则：

```python
x = torch.cat([octa, oct], dim=1)
```

融合后：

```text
[B, 2, H, W]
```

模型改为：

```python
UNet(in_channels=2)
```

### 这一阶段最重要的实验问题

验证：

\[
OCTA + OCT > OCTA
\]

如果简单 concat 都没有明显提升，需要优先检查：

- OCT 与 OCTA 是否真正空间对齐
- 当前 OCT 表示是否适合用于 2D 分割
- OCT 中是否存在对血管分割有帮助的信息
- 模型是否真正利用了 OCT 信息

不要一发现没有提升就直接设计复杂注意力模块。

---

## 5. 最关键的研究难点：3D OCT 与 2D OCTA 分割的对应

这是整个课题最值得研究的地方。

实际数据往往不是：

```text
一张 OCT + 一张 OCTA
```

而更接近：

```text
OCT volume
[D, H, W]

OCTA volume
[D, H, W]

       ↓ projection

OCTA en-face
[H, W]

       ↓ segmentation

Vessel mask
[H, W]
```

因此真正的问题是：

> 如何从 3D structural OCT volume 中提取对 2D OCTA vessel segmentation 有帮助的信息？

这比简单的双通道输入更有研究价值。

需要重点研究：

- OCT axial 方向含义
- retinal layer 分布
- 哪些 OCT depth / layer 对血管有帮助
- 如何将 3D OCT 映射为有意义的 2D structural representation
- 是否需要 learned projection / attention projection

---

## 6. 第三阶段：真正的多模态融合网络

在 early fusion baseline 跑通后，再进入双分支结构。

基本结构：

```text
          OCTA
            ↓
      OCTA Encoder
            ↓
         F_octa
            │
            ├──── Fusion ─── Decoder ─── Vessel Mask
            │
          F_oct
            ↑
       OCT Encoder
            ↑
           OCT
```

数学表示：

\[
F_{OCTA}=E_{OCTA}(X_{OCTA})
\]

\[
F_{OCT}=E_{OCT}(X_{OCT})
\]

\[
F_{fusion}=Fusion(F_{OCTA},F_{OCT})
\]

\[
Mask=D(F_{fusion})
\]

第一版 Fusion 仍然建议直接：

```python
F = torch.cat([F_octa, F_oct], dim=1)
```

然后逐步尝试：

```text
Concat
↓
Channel Attention
↓
Cross Attention
↓
Gated Fusion
↓
Multi-scale Fusion
```

这样天然可以形成完整的消融实验。

---

## 7. 可考虑的创新方向

OCTA 可以提供：

- 血流信息
- 血管显著性
- 毛细血管网络

但容易受到：

- 弱血流
- 低对比度
- 噪声
- 投影伪影
- 血管断裂

影响。

OCT 可以提供：

- 解剖结构信息
- 视网膜层信息
- 组织边界
- 局部反射特征

因此可以把论文的核心思路设计成：

> 利用 OCT 提供的结构先验，辅助 OCTA 恢复细小、断裂和低对比度血管，并提升血管拓扑连续性。

### 可能的方向

#### 方向 1：Structural Prior

由 OCT 分支生成：

```text
Structural Prior Map
```

再用于调制 OCTA 特征：

\[
F'_{OCTA}=F_{OCTA}\odot Attention(F_{OCT})
\]

#### 方向 2：Cross-modal Attention

让 OCTA 特征主动查询 OCT 结构信息：

```text
OCTA Feature
      ↓
Cross Attention ← OCT Feature
      ↓
Enhanced OCTA Feature
```

#### 方向 3：Multi-scale Fusion

在不同 encoder stage 进行多尺度融合：

```text
Stage 1 → Fusion
Stage 2 → Fusion
Stage 3 → Fusion
Stage 4 → Fusion
```

重点观察：

- 小血管
- 血管边界
- 拓扑连续性

是否得到改善。

---

## 8. 推荐实验设计

最终实验表可以按照下面的逻辑设计：

| Method | OCTA | OCT | Dice | IoU | Recall | clDice |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | ✓ |  |  |  |  |  |
| UNet++ | ✓ |  |  |  |  |  |
| Early Fusion | ✓ | ✓ |  |  |  |  |
| Dual Encoder | ✓ | ✓ |  |  |  |  |
| + Attention | ✓ | ✓ |  |  |  |  |
| Your Method | ✓ | ✓ |  |  |  |  |

整体实验逻辑：

```text
OCTA-only baseline
↓
证明 OCT 是否有帮助
↓
证明 dual encoder 是否优于简单拼接
↓
证明 fusion module 是否有效
↓
证明细血管与拓扑连续性是否改善
```

---

## 9. 消融实验建议

后续至少可以做以下消融：

### 模态消融

```text
OCTA only
OCT only
OCTA + OCT
```

### Fusion 消融

```text
Concat
Channel Attention
Cross Attention
Gated Fusion
Multi-scale Fusion
```

### OCT 表示方式消融

例如：

```text
单层 OCT
多层 OCT
平均投影
最大投影
learned projection
3D encoder feature
```

### Loss 消融

例如：

```text
BCE + Dice
BCE + Dice + clDice
Focal + Dice
Topology-aware Loss
```

---

## 10. 当前最优先完成的三件事

### 1. 彻底弄懂 OCTA-500 数据结构

随机选择一个样本，同时可视化：

```text
OCT volume
OCTA volume
OCTA projection
Vessel GT
```

并记录：

```text
shape
dtype
value range
空间方向
对应关系
```

---

### 2. 跑通 OCTA-only U-Net

先得到一个稳定 baseline。

需要完成：

```text
训练
验证
测试
Dice
IoU
Recall
预测结果可视化
```

不要在这一阶段加入复杂创新。

---

### 3. 弄清 OCT 与 OCTA 的空间对应关系

重点确认：

```text
OCT volume shape = ?
OCTA volume shape = ?
projection shape  = ?
GT shape          = ?

axial 方向是哪一维？
en-face 是怎样生成的？
OCT 与 OCTA 是否已经配准？
哪个 retinal layer 最可能提供血管结构信息？
```

这一步比选择 Transformer、Mamba 或 Attention 更重要。

---

# 推荐整体路线

```text
理解数据
   ↓
OCTA-only U-Net baseline
   ↓
OCT + OCTA 简单 concat baseline
   ↓
验证 OCT 是否真正有增益
   ↓
Dual Encoder
   ↓
设计 Fusion Module
   ↓
Structural Prior / Cross Attention
   ↓
Multi-scale Fusion
   ↓
Loss 与 topology 优化
   ↓
消融实验
   ↓
细血管与拓扑连续性分析
```

## 一句话总结

> **先看懂数据，再跑通 baseline，再证明 OCT 有帮助，最后才设计复杂的多模态融合网络。**
