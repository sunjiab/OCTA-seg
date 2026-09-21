# OCTA 实验进度与续做指南

> 更新日期：2026-09-21  
> 项目目录：`/root/workplace/octa_baseline`  
> 当前阶段：OCTA 单模态 U-Net baseline 已完成，准备进入 OCT + OCTA 多模态实验。

## 1. 课题目标与当前结论

课题的最终目标是利用配准的结构 OCT 信息辅助 OCTA 血管分割，重点改善细小血管、低对比度区域、血管断裂以及投影伪影干扰。

当前已经完成研究路线中的第一阶段：

```text
OCTA(ILM_OPL) → U-Net → 二值血管掩膜
```

现有训练、验证、测试、断点保存、指标统计和结果可视化流程均已跑通。除最初的 `GT_Capillary` baseline 外，还在完全相同的实验设置下完成了动脉和静脉二值分割实验。

当前三个测试集实验的 Dice 排序为：

```text
动脉 0.9008 > GT_Capillary 0.8953 > 静脉 0.8810
```

这些结果构成后续 OCT + OCTA 多模态方法的单模态对照基线。

## 2. 数据集认识

使用数据集：OCTA-500。

目前使用的是 3 mm 子集：

- 病例编号：`10301-10500`，共 200 例；
- 图像大小：`304 × 304`；
- 当前输入：`3mm/OCTA(ILM_OPL)` 灰度投影图；
- 当前标签：`GT_Capillary`、`GT_Artery` 或 `GT_Vein`；
- 输入 BMP 按 `0~255 → 0~1` 归一化；
- 标签采用 `mask > 0` 转换为二值掩膜。

固定病例级划分如下：

| 子集 | 病例编号 | 数量 |
|---|---|---:|
| Train | 10301-10440 | 140 |
| Validation | 10441-10450 | 10 |
| Test | 10451-10500 | 50 |

病例之间没有跨集合重复，避免了病例级数据泄漏。

需要保留的语义注意事项：

- 每个病例对每一种标注类型只有一张 2D 标签，并不是每个深度层各有一张相同标签；
- 不同深度 slab 的血管分布不应被理解为完全相同；
- `GT_Artery` 和 `GT_Vein` 分别表示动脉与静脉；
- 当前代码把 `GT_Capillary` 直接作为二值目标使用。从标签密集程度看，它覆盖了较广泛的血管网络。在论文中将其严格表述为“毛细血管”还是“完整/毛细血管网络”前，应再次核对 OCTA-500 官方标签定义。

## 3. 环境与工程状态

Conda 环境已经创建：

```bash
conda activate octa
```

已验证的主要环境：

- Python 3.10.14；
- PyTorch 2.1.0 + CUDA 11.8；
- NVIDIA RTX 3090；
- AMP 混合精度训练可用；
- Jupyter 内核名称：`Python (octa)`；
- 项目已经以 editable 模式安装；
- 基础自动化测试曾通过：`4 passed`。

重新检查代码的命令：

```bash
cd /root/workplace
conda run -n octa pytest -q octa_baseline/tests
```

## 4. 已实现的代码模块

| 文件 | 作用 |
|---|---|
| `train.py` | 参数解析、训练、验证、早停、保存 checkpoint、最终测试 |
| `octa/data.py` | 数据划分、BMP 读取、归一化、配对数据增强、DataLoader |
| `octa/model.py` | 四层 U-Net 编码器、转置卷积解码器、跳跃连接 |
| `octa/losses.py` | BCEWithLogits + Soft Dice 组合损失 |
| `octa/metrics.py` | 基于全测试集混淆矩阵的二分类分割指标 |
| `octa/engine.py` | 单轮训练或验证逻辑 |
| `octa/utils.py` | 随机种子和 JSON 等辅助功能 |
| `tests/` | 数据、模型、损失及训练流程基础测试 |
| `结果与标签可视化.ipynb` | 加载 checkpoint，展示预测、标签、误差图和轮廓图 |

模型为四层 U-Net：

```text
1×304×304
 → 32×304×304
 → 64×152×152
 → 128×76×76
 → 256×38×38
 → 512×19×19
 → 256×38×38
 → 128×76×76
 → 64×152×152
 → 32×304×304
 → 1×304×304 logits
```

解码器使用 `2×2, stride=2` 转置卷积上采样，再与同尺度编码器特征按通道拼接，最后用 DoubleConv 融合。网络共有约 776 万个可训练参数。

## 5. 三个实验的共同设置

| 设置 | 数值 |
|---|---|
| 输入 | `OCTA(ILM_OPL)` |
| 扫描范围 | 3 mm |
| 输入尺寸 | `1 × 304 × 304` |
| 网络 | U-Net，`base_channels=32` |
| Epoch 上限 | 100 |
| Batch size | 8 |
| DataLoader workers | 4 |
| 优化器 | AdamW |
| 初始学习率 | 0.001 |
| Weight decay | 0.0001 |
| Scheduler | ReduceLROnPlateau，factor=0.5，patience=5 |
| 早停 | 验证 Dice 连续 20 轮不提升 |
| Loss | BCEWithLogits + Soft Dice，权重 1:1 |
| 二值化阈值 | 0.5 |
| 随机种子 | 42 |
| AMP | 开启 |
| 最优模型标准 | 最大验证集 Dice |

训练增强只用于训练集：随机旋转 `0°/90°/180°/270°`、水平翻转和垂直翻转。验证集与测试集不做随机增强。

指标采用整个数据集累计 TP、FP、FN、TN 后计算的 micro 指标，而不是先算每个病例再求平均。

## 6. 三类标签实验结果

### 6.1 测试集最终结果

| 目标标签 | Best epoch | Test loss | Dice | IoU | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GT_Capillary` | 57 | 0.3459 | 0.8953 | 0.8104 | 0.8728 | **0.9190** | 0.9043 |
| `GT_Artery` | 70 | 0.1425 | **0.9008** | **0.8194** | **0.8947** | 0.9069 | 0.9958 |
| `GT_Vein` | 81 | 0.1586 | 0.8810 | 0.7873 | 0.8895 | 0.8727 | **0.9968** |

### 6.2 最佳 checkpoint 对应的验证集结果

| 目标标签 | Epoch | Val loss | Dice | IoU | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GT_Capillary` | 57 | 0.3309 | **0.9028** | **0.8228** | 0.8864 | **0.9198** | 0.9083 |
| `GT_Artery` | 70 | 0.1450 | 0.8979 | 0.8148 | **0.8956** | 0.9003 | 0.9963 |
| `GT_Vein` | 81 | 0.1489 | 0.8883 | 0.7990 | 0.8944 | 0.8823 | **0.9970** |

### 6.3 最佳 checkpoint 对应的训练集结果

| 目标标签 | Train loss | Dice | IoU | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| `GT_Capillary` | 0.3403 | 0.8973 | 0.8138 | 0.8837 | 0.9114 | 0.9123 |
| `GT_Artery` | 0.1238 | **0.9132** | **0.8402** | **0.9179** | **0.9085** | 0.9969 |
| `GT_Vein` | 0.1341 | 0.9008 | 0.8194 | 0.9056 | 0.8960 | **0.9972** |

### 6.4 训练停止情况

| 目标标签 | 实际训练轮数 | 最佳轮次 | 停止原因 |
|---|---:|---:|---|
| `GT_Capillary` | 77 | 57 | 早停 |
| `GT_Artery` | 90 | 70 | 早停 |
| `GT_Vein` | 100 | 81 | 达到最大 epoch |

## 7. 当前结果的解释

- 动脉模型综合结果最好，测试 Dice 为 `0.9008`；
- `GT_Capillary` 模型 Recall 最高，漏检较少，但 Precision 略低，误检相对更多；
- 静脉模型 Dice 最低，主要表现为 Recall 较低，即静脉漏检更多；
- 三个模型的验证集和测试集 Dice 接近，暂未看到明显的泛化崩塌；
- 动脉和静脉标签稀疏、背景占比很大，因此 Specificity 接近 1，不能据此单独判断血管分割质量；
- 不同标签的前景比例明显不同，BCE 部分受到类别比例影响，因此三类任务的 Loss 数值不宜直接横向排名；
- 横向比较应优先看 Dice、IoU，并结合 Precision 与 Recall 判断误检和漏检。

## 8. 可视化已经完成

每个实验目录都有自动生成的 `test_predictions.png`。此外，`结果与标签可视化.ipynb` 可以生成更详细的六列对比：

```text
OCTA input
Ground truth
Probability map
Binary prediction (threshold=0.5)
Error map
Contour overlay
```

误差图颜色定义：

- 绿色：TP，真实血管且预测正确；
- 红色：FP，背景被误判为血管；
- 蓝色：FN，真实血管被漏检；
- 黑色：TN，背景预测正确。

轮廓图中的 Dice 和 IoU 是当前病例的指标，不是整个测试集指标。查看不同实验时，需要修改 Notebook 顶部的 `RUN_DIR`：

```python
RUN_DIR = PROJECT_ROOT / "runs/unet_3mm_artery"
```

可替换为：

```text
runs/unet_3mm_ilm_opl
runs/unet_3mm_artery
runs/unet_3mm_vein
```

## 9. 实验产物位置

### GT_Capillary

```text
runs/unet_3mm_ilm_opl/
├── config.json
├── history.csv
├── best.pt
├── last.pt
├── test_metrics.json
├── test_predictions.png
└── tensorboard/
```

### 动脉

```text
runs/unet_3mm_artery/
├── config.json
├── history.csv
├── best.pt
├── last.pt
├── test_metrics.json
├── test_predictions.png
└── tensorboard/
```

### 静脉

```text
runs/unet_3mm_vein/
├── config.json
├── history.csv
├── best.pt
├── last.pt
├── test_metrics.json
├── test_predictions.png
└── tensorboard/
```

每个 `best.pt` 和 `last.pt` 约 89 MiB。后续不要覆盖这些目录；新实验必须使用新的 `--output-dir`。

查看三组训练曲线：

```bash
cd /root/workplace
conda run -n octa tensorboard --logdir octa_baseline/runs
```

## 10. 当前阶段完成清单

- [x] 理解 OCTA-500 的主要目录和标签类型；
- [x] 完成 3 mm OCTA 单模态数据加载；
- [x] 按病例划分训练、验证和测试集；
- [x] 实现配对几何数据增强；
- [x] 实现四层 U-Net；
- [x] 实现 BCE + Dice Loss；
- [x] 实现 Dice、IoU、Precision、Recall、Specificity；
- [x] 实现 AMP、学习率调度、早停和 checkpoint；
- [x] 完成 `GT_Capillary` 训练与正式测试；
- [x] 完成动脉训练与正式测试；
- [x] 完成静脉训练与正式测试；
- [x] 完成预测、标签、误差和轮廓可视化；
- [x] 编写单模态 baseline 详细总结；
- [ ] 确认 `GT_Capillary` 的官方精确定义；
- [ ] 加入 clDice 或血管拓扑指标；
- [ ] 完成 OCT-only 对照；
- [ ] 完成 OCT + OCTA early-fusion baseline；
- [ ] 完成双编码器及后续融合方法。

## 11. 下一阶段建议：先做最简单的多模态 baseline

下一步建议严格保持当前数据划分、标签、增强、优化器和训练参数不变，只加入 OCT 模态，首先回答：

> 相同 U-Net 和相同实验条件下，OCT + OCTA 是否优于 OCTA-only？

建议按以下顺序进行。

### 步骤 1：确认 OCT 与 OCTA 的空间对应关系

随机抽取若干病例，同时可视化并记录：

- OCT 投影、OCTA 投影和目标标签的 shape；
- 图像方向是否一致；
- 血管或明显结构位置是否对齐；
- 是否需要翻转、旋转或 resize；
- OCT 采用 FULL、ILM_OPL、OPL_BM 中哪一种表示。

在空间对应没有确认前，不建议直接训练复杂融合模型。

### 步骤 2：建立 OCT-only 对照

先让单通道 U-Net 只输入 OCT 投影，并保持其他设置不变。该实验用于判断结构 OCT 本身包含多少可用于预测血管标签的信息。

建议输出目录：

```text
runs/unet_3mm_oct_ilm_opl
```

### 步骤 3：实现双通道 early fusion

数据加载器同时读取 OCTA 和 OCT，并在通道维拼接：

```python
x = torch.cat([octa, oct], dim=0)  # 单样本得到 [2, H, W]
```

批处理后为：

```text
[B, 2, 304, 304]
```

模型已经支持 `UNet(in_channels=2)`，但当前 `train.py` 使用的是默认单通道初始化，因此需要把 `in_channels` 暴露为训练参数并随 checkpoint 保存。

主要修改入口：

- `octa/data.py`：同时读取并同步增强 OCTA、OCT 和 mask；
- `train.py`：增加 OCT 路径/表示方式和 fusion 参数；
- `train.py`：early fusion 时构造 `UNet(in_channels=2, base=...)`；
- `结果与标签可视化.ipynb`：增加 OCT 输入列；
- `tests/`：增加双通道形状、同步增强和 forward 测试。

建议输出目录：

```text
runs/unet_3mm_octa_oct_early_fusion
```

### 步骤 4：公平对比

首轮多模态实验优先使用 `GT_Capillary`，因为它是当前主 baseline 的目标。至少比较：

| Method | OCTA | OCT | Dice | IoU | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net baseline | ✓ |  | 0.8953 | 0.8104 | 0.8728 | 0.9190 | 0.9043 |
| OCT-only U-Net |  | ✓ | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| Early fusion U-Net | ✓ | ✓ | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

只有在简单拼接的收益明确、数据对齐确认无误后，再进入 dual encoder、attention 或 multi-scale fusion。

## 12. 后续实验注意事项

1. 不要用测试集选择阈值、checkpoint 或模型结构；当前测试集应只用于最终评估。
2. 当前验证集只有 10 例，后续方法差距较小时建议补充多随机种子实验，报告均值和标准差。
3. 新方法必须复用相同病例划分，否则指标不能公平比较。
4. 训练新目标或新模态时必须使用新输出目录，避免覆盖已有模型。
5. 如果改变标签前景比例，不能只看 Accuracy 或 Specificity。
6. 血管任务建议后续增加 clDice，并检查细血管连通性、断裂和最差病例。
7. 目前三个实验是三个独立二分类模型，不是一个同时区分毛细血管、动脉和静脉的多分类模型。
8. 如果后续改为多分类，需要重构标签编码、输出通道数、损失函数和评估逻辑。

## 13. 下次继续时的快速入口

```bash
cd /root/workplace
conda activate octa

# 检查当前代码
pytest -q octa_baseline/tests

# 查看历史训练曲线
tensorboard --logdir octa_baseline/runs
```

建议下次首先打开：

1. `OCTA实验进度与续做指南.md`：查看总体进度；
2. `OCTA单模态Baseline实验总结.md`：查看主 baseline 的完整细节；
3. `OCTA_OCT_多模态血管分割研究路线.md`：查看整体研究路线；
4. `octa/data.py`：开始改造双模态数据加载；
5. `train.py`：增加 early-fusion 参数和双通道模型初始化。

下一次工作的推荐起点是：**先写一个 OCT/OCTA 配准与对应关系检查 Notebook，再实现 OCT-only 和双通道 early-fusion baseline。**
