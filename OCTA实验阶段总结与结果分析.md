# OCTA 实验阶段总结与结果分析

记录日期：2026-09-24。根据当前工作区的 `config.json`、`history.csv`、`test_metrics.json`、`val_metrics.json`、`ablation_metrics.json`、逐病例 CSV 和实现代码核对。四个 UGR 最佳 checkpoint 均已完成 50 例测试集评估。

## 1. 当前完成情况与主要结论

目前已完成 **16 组非 smoke 训练**：原 OCTA 单模态 3 组、OCT-only 与 Early Fusion 6 组、OCT 动脉延长训练 1 组、UGR 四类标签 4 组，以及大血管同入口 OCTA/Early 对照 2 组。四个 UGR 模型另外完成了正常、OCT 置零和三次 OCT 病例打乱的测试集推理消融。

- 16 组非 smoke 训练均已有测试结果；大血管同入口 OCTA/Early 已复用最佳权重补齐测试，无需重新训练。
- OCTA 是当前有效的主要输入。简单输入级融合的历史测试收益很小，同时表现为 Precision 上升、Recall 下降。
- UGR 测试 Dice 为：毛细血管 **0.895567**、动脉 **0.904738**、静脉 **0.882617**、大血管 **0.920490**。前三类均高于历史 OCTA 和 Early，其中动脉增量最大。
- 相对历史 Early，UGR 测试 Dice 分别增加 0.000134、0.002870、0.000925；只有动脉的提升达到约 0.287 个百分点，其余仍很小。
- 最可比的大血管同入口对照中，UGR Dice **0.923367**、Early **0.923335**，差距仅 **0.00003187（0.003187 个百分点）**，当前应视为基本持平。
- OCT 打乱后的 Dice 几乎不变，表明当前 UGR 对正确病例配对的 OCT 依赖很弱。更高分不能直接解释为 OCT 提供了稳定有效的信息。
- 所有非 smoke 结果只有 seed42。测试集已经参与多轮方案分析，尚未建立多随机种子稳定性、全套机制消融或外部泛化证据。

旧 UGR README 和验证记录中的“仅完成一轮 smoke、正式训练待完成”已经滞后；截至本文，四类 UGR 和大血管对照均已完成训练。工程验证记录仍可作为代码验收历史。

## 2. 数据与评估协议

| 项目 | 当前设置 |
|---|---|
| 数据 | OCTA-500，`data/OCTA-500数据集`，3 mm 子集 |
| 输入 | 配对的 `OCTA(ILM_OPL)` 与 `OCT(ILM_OPL)` 二维 en-face 投影 |
| 空间尺寸 | 3 mm 投影为 304×304；读取时保留原尺寸 |
| 训练集 | 10301–10440，140 例 |
| 验证集 | 10441–10450，10 例 |
| 测试集 | 10451–10500，50 例 |
| 标签 | GT_Capillary、GT_Artery、GT_Vein、GT_LargeVessel |
| 任务形式 | 每类单独训练一个二分类模型；不是联合多分类 |
| 输入预处理 | 灰度，浮点数除以 255 |
| 标签预处理 | 对应 BMP 中像素大于 0 作为前景 |
| 增强 | 配对输入与标签同步旋转/翻转，仅训练集启用 |
| 模型选择 | 验证 micro Dice 最佳的 `best.pt` |
| 判定阈值 | sigmoid 概率 ≥0.5 为前景 |

GT_LargeVessel 直接使用数据集提供的标签，并非代码临时计算动脉与静脉并集。GT_Capillary 是目录名称；若要论述其是否严格排除大血管，应进一步核查数据集标注定义，不能仅由名称断定。

指标先累计整个划分的 TP、FP、FN、TN，再计算，因此是 **micro 指标**，不是每例 Dice 的简单平均：

```text
Dice        = 2TP / (2TP + FP + FN)
IoU         = TP / (TP + FP + FN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

Specificity 高只能说明背景识别较好。血管很稀疏时，即使漏掉大量血管，它也可能接近 1。不同标签前景比例不同，不宜单纯跨标签排名任务难度。

## 3. 模型和训练设置

| 方法 | 输入与结构 | 参数量 |
|---|---|---:|
| OCTA-only | 单通道 OCTA → U-Net | 7,762,465 |
| OCT-only | 单通道 OCT 投影 → U-Net | 7,762,465 |
| Early Fusion | OCTA/OCT 通道拼接 → 双通道 U-Net | 7,762,753 |
| UGR-Fusion | OCTA U-Net 锚点 + OCT 轻量金字塔 + 五尺度门控残差修正 | 7,919,371 |

UGR 比 OCTA-only 增加 156,906 个参数，约 2.02%；这不等于推理时间或计算量也仅增加 2.02%。

共同参数：AdamW，学习率 0.001，weight decay 0.0001，batch size 8，workers 4，base channels 32，AMP 开启，seed42。最大 100 轮，验证 Dice 连续 20 轮未提高则早停；ReduceLROnPlateau 的 factor=0.5、patience=5。OCT 动脉扩展组仅将最大轮数改为 200。

UGR 的实际结构以 [model.py](octa_ugr_fusion/model.py) 为准：

1. OCTA U-Net 的五尺度通道为 32/64/128/256/512，产生独立 anchor logits。
2. OCT 金字塔通道为 8/16/32/64/128；层间用平均池化降采样，块内为普通 3×3 卷积、深度 3×3 卷积、GroupNorm 和 SiLU。
3. 每个融合头将两模态投影至各 16 通道，拼接两特征、绝对差与 1 通道不确定性，共 49 通道；3×3 卷积融合至 16 通道。
4. 门控与残差输出层均为 **1×1 卷积**。门控 sigmoid 后与有符号残差相乘；五尺度上采样后取**平均**。
5. 聚合后统一施加 tanh 幅度限制与不确定性权重，再加到 anchor logits。

```text
p_A = sigmoid(z_A).detach()
u = 4 p_A (1-p_A)
r = mean_l[Upsample(g_l × d_l)]
delta = strength × (0.1 + 0.9u) × 2 × tanh(r/2)
z_final = z_A + delta
```

训练前 5 轮线性增加 strength；验证/推理使用 1。不确定性是概率接近 0.5 的启发式度量，并非校准后的错误概率。残差输出层零初始化；这表示初始化时与模型内部 anchor 等价，不表示加载了已训练的旧 baseline。

基础损失是 BCEWithLogits + Soft Dice。UGR 为：

```text
L = L_seg(final) + 0.3 L_seg(anchor) + 0.1 L_keep
```

L_keep 只在标签前景且 anchor 概率≥0.5 的位置，惩罚最终概率低于已 detach 的 anchor 概率。它是软约束，不能保证 Recall 一定提高。四类正式 UGR 配置的 clDice 权重均为 0，因此当前收益不能归因于拓扑损失。不同方法的总 loss 含不同项，横向比较应使用同定义的 `seg_loss` 与分割指标。

## 4. 历史测试集结果：OCTA / OCT / Early Fusion

下表均来自 50 例测试集上的最佳验证 checkpoint。数值保留六位，实际计算依据未舍入结果。

| 标签 | 方法 | Dice | IoU | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|
| Capillary | OCTA | 0.895284 | 0.810419 | 0.872786 | 0.918971 | 0.904318 |
| Capillary | OCT | 0.620663 | 0.449972 | 0.475929 | 0.891896 | 0.298443 |
| Capillary | Early | 0.895433 | 0.810664 | 0.878237 | 0.913315 | 0.909547 |
| Artery | OCTA | 0.900764 | 0.819445 | 0.894678 | 0.906933 | 0.995778 |
| Artery | OCT | 0.637297 | 0.467671 | 0.725746 | 0.568065 | 0.991510 |
| Artery | Early | 0.901868 | 0.821275 | 0.907126 | 0.896671 | 0.996369 |
| Vein | OCTA | 0.881008 | 0.787323 | 0.889497 | 0.872679 | 0.996756 |
| Vein | OCT | 0.018257 | 0.009213 | 0.034807 | 0.012374 | 0.989734 |
| Vein | Early | 0.881692 | 0.788416 | 0.894264 | 0.869469 | 0.996924 |
| Artery | OCT，最大200轮 | 0.638381 | 0.468840 | 0.728397 | 0.568167 | 0.991622 |

| 标签 | 方法 | 最佳轮次 / 实际轮数 | 最佳 Val Dice | Test loss |
|---|---|---:|---:|---:|
| Capillary | OCTA | 57 / 77 | 0.902784 | 0.345927 |
| Capillary | OCT | 3 / 23 | 0.639978 | 1.170264 |
| Capillary | Early | 55 / 75 | 0.902349 | 0.344599 |
| Artery | OCTA | 70 / 90 | 0.897948 | 0.142514 |
| Artery | OCT | 97 / 100 | 0.659858 | 0.522747 |
| Artery | Early | 78 / 98 | 0.900539 | 0.140342 |
| Vein | OCTA | 81 / 100 | 0.888298 | 0.158644 |
| Vein | OCT | 1 / 21 | 0.017526 | 1.546604 |
| Vein | Early | 88 / 100 | 0.882347 | 0.155432 |
| Artery | OCT，最大200轮 | 107 / 127 | 0.660581 | 0.521299 |

Early 相对 OCTA 的测试 Dice 增量分别为 +0.000149、+0.001104、+0.000684，即 +0.0149、+0.1104、+0.0684 个百分点。三类均伴随 Precision 上升和 Recall 下降，符合预测更保守的表现，但不能据此确定网络内部原因。

OCT 毛细血管结果呈现高 Recall、低 Precision 和低 Specificity，说明误检严重；OCT 动脉存在一定分割能力，但漏检明显。延长动脉训练仅带来约 0.001085 的测试 Dice 增量，未解决主要性能差距。OCT 静脉训练最后验证 Dice 为 0，历史最佳第 1 轮的测试 Dice 仍仅 0.018257。当前配置存在明显失败，不能扩展为“OCT在任何条件下都没有静脉信息”。

## 5. UGR 验证、测试与模态扰动结果

### 5.1 验证集结果

以下为 **10 例验证集**。前三类的 OCTA/Early 是历史入口，大血管三种模型来自同一新入口。

| 标签 | 方法 | 最佳轮次 / 实际轮数 | Dice | IoU | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| Capillary | UGR | 44 / 64 | 0.902568 | 0.822436 | 0.890285 | 0.915194 | 0.912245 |
| Artery | UGR | 87 / 100 | 0.902592 | 0.822475 | 0.901550 | 0.903636 | 0.996512 |
| Vein | UGR | 64 / 84 | 0.886268 | 0.795763 | 0.886199 | 0.886336 | 0.996735 |
| LargeVessel | OCTA，同入口 | 47 / 67 | 0.922285 | 0.855778 | 0.927424 | 0.917202 | 0.995228 |
| LargeVessel | Early，同入口 | 71 / 91 | 0.923335 | 0.857588 | 0.924210 | 0.922461 | 0.994971 |
| LargeVessel | UGR | 80 / 100 | 0.923367 | 0.857643 | 0.927874 | 0.918903 | 0.995251 |

| 标签/方法 | 总 loss | seg_loss | anchor loss | retention loss | gate_mean | correction_abs |
|---|---:|---:|---:|---:|---:|---:|
| Capillary / UGR | 0.430169 | 0.330409 | 0.331571 | 0.002891 | 0.615942 | 0.083871 |
| Artery / UGR | 0.179475 | 0.137427 | 0.139475 | 0.002060 | 0.850548 | 0.158147 |
| Vein / UGR | 0.202013 | 0.154370 | 0.157898 | 0.002736 | 0.876432 | 0.162972 |
| LargeVessel / OCTA | 0.135136 | 0.135136 | 0 | 0 | — | — |
| LargeVessel / Early | 0.123877 | 0.123877 | 0 | 0 | — | — |
| LargeVessel / UGR | 0.168117 | 0.128255 | 0.131780 | 0.003283 | 0.863941 | 0.172871 |

`correction_abs` 是平均绝对 logit 修正，不是概率变化或 Dice 增量。较大的 gate 不能证明 OCT 有用；较小的平均修正也不代表没有边界影响。需要结合模型内部 anchor/final 配对指标、OCT 打乱和容量对照。

### 5.2 UGR 正常输入的测试集结果

四个 UGR 均使用各自验证集 Dice 最佳的 checkpoint，在 50 例测试集上评估：

| 标签 | Best epoch | Dice | IoU | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Capillary | 44 | 0.895567 | 0.810883 | 0.877159 | 0.914764 | 0.908488 |
| Artery | 87 | **0.904738** | **0.826047** | 0.901490 | **0.908009** | 0.996076 |
| Vein | 64 | 0.882617 | 0.789896 | 0.891214 | 0.874184 | 0.996807 |
| LargeVessel | 80 | **0.920490** | **0.852692** | **0.926511** | **0.914547** | **0.994747** |

不同标签任务不能直接横向排名。前三类可与历史测试结果对比；大血管同入口完整测试对照见第 6.3 节。

### 5.3 OCT 置零与病例打乱

| 标签 | 正常 OCT Dice | OCT置零 | 置零变化 | 打乱OCT均值 | 打乱变化 |
|---|---:|---:|---:|---:|---:|
| Capillary | 0.895567 | 0.895372 | -0.000195 | 0.895230 | -0.000337 |
| Artery | 0.904738 | 0.904055 | -0.000682 | 0.904613 | -0.000125 |
| Vein | 0.882617 | 0.882063 | -0.000554 | 0.882070 | -0.000547 |
| LargeVessel | 0.920490 | 0.920432 | -0.000058 | 0.920488 | -0.000002 |

打乱均值为 shuffle seed42/43/44 三次 micro Dice 的算术平均。“变化”是扰动值减正常值，因此负值表示性能下降。

逐病例检查与 micro 结果方向基本一致：

| 标签 | 正常优于置零的病例数/50 | 正常优于打乱均值的病例数/50 | 正常−打乱的逐病例 Dice 均值 |
|---|---:|---:|---:|
| Capillary | 30 | 44 | +0.000338 |
| Artery | 34 | 30 | +0.000124 |
| Vein | 30 | 38 | +0.000566 |
| LargeVessel | 26 | 22 | -0.000006 |

毛细血管和静脉对正确配对 OCT 有方向一致但极小的依赖；动脉正常输入优于置零较多，却对病例打乱不敏感；大血管对打乱 OCT 基本完全不敏感。置零 OCT 属于分布外输入，置零下降不能单独证明 OCT 有用。打乱结果更直接地表明：当前辅助分支对“这个病例自己的 OCT”利用有限。

## 6. 当前结果的具体判断

### 6.1 验证集探索性比较

| 标签 | 历史 OCTA Val Dice | 历史 Early Val Dice | UGR Val Dice | UGR−OCTA | UGR−Early |
|---|---:|---:|---:|---:|---:|
| Capillary | 0.902784 | 0.902349 | 0.902568 | -0.000217 | +0.000218 |
| Artery | 0.897948 | 0.900539 | 0.902592 | +0.004643 | +0.002052 |
| Vein | 0.888298 | 0.882347 | 0.886268 | -0.002031 | +0.003920 |

三类数据划分相同，但旧、新训练入口随机数使用与数据加载顺序存在差别；这些结果是历史参考，不能代替同入口匹配对照。

毛细血管三者非常接近，没有明显收益信号。动脉相对历史 Early 的 Recall 从 0.897613 升至 0.903636，Precision 从 0.903484 降至 0.901550，Dice 同时提高，提示减少漏检的设计方向值得进一步检验。

静脉相对历史 Early 的 Recall 从 0.871784 升至 0.886336，但 Precision 从 0.893170 降至 0.886199；相对 OCTA 虽然 Recall 更高，Dice 仍更低。其最后一轮 Val Dice 0.877689 低于最佳 0.886268，说明早停后期出现退化，正式结果必须使用 `best.pt`。

### 6.2 三类任务的完整测试比较

| 标签 | OCTA Dice | Early Dice | UGR Dice | UGR−OCTA | UGR−Early |
|---|---:|---:|---:|---:|---:|
| Capillary | 0.895284 | 0.895433 | **0.895567** | +0.000283 | +0.000134 |
| Artery | 0.900764 | 0.901868 | **0.904738** | +0.003974 | +0.002870 |
| Vein | 0.881008 | 0.881692 | **0.882617** | +0.001609 | +0.000925 |

三类测试 Dice 中 UGR 都最高，但增量尺度不同。毛细血管相对 Early 只提高 0.0134 个百分点，几乎可以视为持平；静脉提高 0.0925 个百分点，也属于很小差异；动脉提高 0.2870 个百分点，是当前最明确的正向结果。

毛细血管相对历史 OCTA 的 Precision 从 0.872786 升至 0.877159，Recall 从 0.918971 降至 0.914764，仍表现为用少量漏检换取更少误检。动脉相对 OCTA 的 Precision 从 0.894678 升至 0.901490、Recall 从 0.906933 升至 0.908009，两项同时提高；相对 Early 则是 Recall 明显提高、Precision 降低。静脉相对 OCTA 的 Precision 与 Recall 都有小幅提高，相对 Early 则主要是 Recall 提高。

测试结果说明 UGR 训练方案在 seed42 上优于历史基线，尤其是动脉；但旧、新入口的随机数消费和 DataLoader 过程不完全一致。加上当前只有一次训练，不能把差值全部归因于 UGR 模块。

### 6.3 大血管是当前最直接的训练对照

UGR 相对 OCTA 的 Val Dice 增加 0.001082（0.1082 个百分点），相对 Early 仅增加 0.000032。与 Early 比较，UGR Precision 增加约 0.003664、Recall 减少约 0.003558；因此大血管任务并未显示预期的召回优势。

Early 的验证 seg_loss 0.123877 低于 UGR 的 0.128255，尽管后者验证 Dice 略高。两者衡量连续概率与阈值化预测的不同方面。

两组大血管对照已复用最佳权重完成 50 例测试：

| 方法 | 最佳轮次 / 实际轮数 | Test Dice | IoU | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| OCTA-only | 47 / 67 | 0.919842 | 0.851581 | 0.926627 | 0.913156 | 0.994764 |
| Early Fusion | 71 / 91 | 0.919732 | 0.851393 | 0.924915 | 0.914608 | 0.994624 |
| UGR | 80 / 100 | 0.920490 | 0.852692 | 0.926511 | 0.914547 | 0.994747 |

UGR 相对 OCTA 测试 Dice 增加 0.000648（0.0648 个百分点），相对 Early 增加 0.000758（0.0758 个百分点）。Early 相对 OCTA 下降 0.000110。三者差异很小，单 seed 不足以证明稳定提升。UGR 相对 Early 主要是 Precision 提高，Recall 基本持平且略低。

Early 的 OCT 置零 Dice 为 0.894677，下降约 0.025055；打乱 OCT 的三次平均 Dice 为 0.919720，仅下降约 0.000012。置零造成的分布变化影响明显，但正确配对信息的作用仍很弱。OCTA-only 的五种推理条件结果完全相同，符合其不读取 OCT 的实现。

### 6.4 UGR 是否真正使用了 OCT

四类正常输入相对三次打乱 OCT 的 Dice 优势均小于 0.00055，大血管差异接近零。这个结果与“UGR 依赖正确配对 OCT 进行明显病例特异修正”的预期不符。

当前较合理的解释包括：OCT 分支贡献很小；模型主要学习不依赖具体 OCT 内容的修正先验；或者 en-face OCT 投影与标签有关的信息不足。打乱实验不能在这些解释之间作最终区分，但已经提示：**不能把 UGR 相对旧基线的全部增益表述为多模态信息增益。**

下一步最关键的不是继续微调门控，而是补容量控制和同入口 OCTA 对照。若辅助分支输入 OCTA 或随机噪声也获得相似结果，增益更可能来自额外容量、辅助监督或优化差异；只有正常配对 OCT 稳定优于这些对照，才能支持 OCT 的有效贡献。

### 6.5 结论边界

- 四个 UGR 和大血管 OCTA/Early 均已有测试及模态扰动结果；毛细血管、动脉、静脉尚缺同入口 OCTA/Early 训练对照。
- 验证集只有 10 例，且用于 checkpoint 选择和方案开发；测试集也已被反复查看，当前报告应视为探索性结果总结。
- 当前正式权重尚无去门控、去不确定性、去保持约束和容量控制训练消融。
- 所有正式实验只有 seed42，微小差异没有均值、标准差或训练级置信区间支持。
- 正式 UGR 没有启用 clDice，也没有二值骨架指标，不能宣称改善了拓扑连通性。
- 病例编号划分并不自动证明患者级无交叉；若存在同患者多眼/多扫描，应核查元数据。
- 更强的泛化结论需要冻结方案后的新独立留出或外部数据验证。

## 7. 可视化成果与图示注意事项

已保存旧实验的 `test_predictions.png`、UGR 的 `val_predictions.png`，以及四类测试消融的 `predictions_and_gates.png`。UGR 图的八列为：OCTA、OCT、标签、anchor 概率、最终预测、误差图、平均 gate、有符号 logit 修正。误差图绿色是真阳性、红色是假阳性、蓝色是假阴性；修正图正值提高前景 logit，负值降低。

这些是展示样本，不是所有病例的视觉评估；本文未对每例图像逐一复核，不据此声称细血管断裂或边界已经改善。

- [毛细血管 UGR 验证图](octa_ugr_fusion/runs/ugr_GT_Capillary_seed42/val_predictions.png)
- [动脉 UGR 验证图](octa_ugr_fusion/runs/ugr_GT_Artery_seed42/val_predictions.png)
- [静脉 UGR 验证图](octa_ugr_fusion/runs/ugr_GT_Vein_seed42/val_predictions.png)
- [大血管 OCTA 验证图](octa_ugr_fusion/runs/octa_GT_LargeVessel_seed42/val_predictions.png)
- [大血管 Early 验证图](octa_ugr_fusion/runs/early_GT_LargeVessel_seed42/val_predictions.png)
- [大血管 UGR 验证图](octa_ugr_fusion/runs/ugr_GT_LargeVessel_seed42/val_predictions.png)
- [毛细血管 UGR 测试解释图](octa_ugr_fusion/runs/ugr_GT_Capillary_seed42_test/predictions_and_gates.png)
- [动脉 UGR 测试解释图](octa_ugr_fusion/runs/ugr_GT_Artery_seed42_test/predictions_and_gates.png)
- [静脉 UGR 测试解释图](octa_ugr_fusion/runs/ugr_GT_Vein_seed42_test/predictions_and_gates.png)
- [大血管 UGR 测试解释图](octa_ugr_fusion/runs/ugr_GT_LargeVessel_seed42_test/predictions_and_gates.png)
- [大血管 OCTA 测试图](octa_ugr_fusion/runs/octa_GT_LargeVessel_seed42_test/predictions_and_gates.png)
- [大血管 Early 测试图](octa_ugr_fusion/runs/early_GT_LargeVessel_seed42_test/predictions_and_gates.png)
- [旧融合交互 Notebook](octa_oct_fusion/结果与标签可视化.ipynb)
- [UGR 中文结构与损失图说明](octa_ugr_fusion/README.md)

**此前 AI 生成结构图存在技术偏差，不宜直接用作论文中的准确方法图。** 当前真实输入是同平面的 OCT en-face 投影，而生成图画了 OCT 横截面；生成图还把平均池化写成步长卷积、将 1×1 输出头画成 3×3、把多尺度平均画成求和，并把有界化位置画得不准确。其影像是生成示意，不是真实病例原图。后续应按本报告第 3 节及源代码修正，再用于正式展示。

## 8. 后续实验优先级与续做命令

1. **补齐同入口对照**：毛细血管、动脉、静脉各补 OCTA 与 Early；大血管三组训练及测试已齐全。优先动脉，因为目前提升信号最明确。
2. **做容量与机制消融**：辅助分支输入 OCTA 的容量对照、关闭不确定性、关闭门控、关闭保持约束。打乱 OCT 几乎不降后，这一步比继续调门控参数更关键。
3. **比较 anchor 与 final**：在同一个 UGR checkpoint 内逐病例计算 anchor/final Dice 与 TP、FP、FN 差异，确认残差修正本身改善了哪些像素。
4. **重复种子**：以 seed42/43/44 完成同协议训练，报告均值和标准差；逐病例配对差值不能替代训练随机性分析。
5. **独立验证**：当前测试集已被查看并影响研究判断，后续冻结方案后应保留新的内部划分或使用外部数据。

以下为命令参考：大血管两条测试命令已执行，重跑需更换输出目录。其他训练命令未在本次执行。

```bash
cd /root/workplace
conda activate octa

# 同入口动脉对照；其他标签替换 GT_Artery
python octa_ugr_fusion/train.py --model octa --target GT_Artery --skip-test
python octa_ugr_fusion/train.py --model early --target GT_Artery --skip-test

# 补齐大血管对照的测试结果；无需重新训练
python octa_ugr_fusion/evaluate.py \
  --run-dir octa_ugr_fusion/runs/octa_GT_LargeVessel_seed42 \
  --split test --output-dir octa_ugr_fusion/runs/octa_GT_LargeVessel_seed42_test
python octa_ugr_fusion/evaluate.py \
  --run-dir octa_ugr_fusion/runs/early_GT_LargeVessel_seed42 \
  --split test --output-dir octa_ugr_fusion/runs/early_GT_LargeVessel_seed42_test

# 容量控制：辅助分支也读取 OCTA
python octa_ugr_fusion/train.py --target GT_Artery --aux-source octa --skip-test \
  --output-dir octa_ugr_fusion/runs/artery_capacity_seed42

# 新种子独立训练
python octa_ugr_fusion/train.py --target GT_Artery --seed 43 --skip-test \
  --output-dir octa_ugr_fusion/runs/ugr_GT_Artery_seed43

```

每次使用新的输出目录。UGR 入口没有 resume 参数，重新调用 train.py 是重新训练；`evaluate.py` 则直接载入 best.pt，生成 `ablation_metrics.json` 和逐病例 CSV。

## 9. 结果来源索引

每个目录的配置记录训练参数，history 记录逐轮训练/验证指标；本报告的评估表读取对应指标 JSON，最佳轮次同时与 history 核对。

| 实验 | 目录 |
|---|---|
| 旧 OCTA 毛细血管 | `octa_baseline/runs/unet_3mm_ilm_opl` |
| 旧 OCTA 动脉 | `octa_baseline/runs/unet_3mm_artery` |
| 旧 OCTA 静脉 | `octa_baseline/runs/unet_3mm_vein` |
| 旧 OCT-only 三类 | `octa_oct_fusion/runs/oct_3mm_ILM_OPL_GT_{Capillary,Artery,Vein}_seed42` |
| 旧 Early 三类 | `octa_oct_fusion/runs/fusion_3mm_ILM_OPL_GT_{Capillary,Artery,Vein}_seed42` |
| OCT 动脉扩展 | `octa_oct_fusion/runs/oct__3mm_ILM_OPL_GT_artery_epochs200` |
| UGR 四类 | `octa_ugr_fusion/runs/ugr_GT_{Capillary,Artery,Vein,LargeVessel}_seed42` |
| UGR 四类测试消融 | `octa_ugr_fusion/runs/ugr_GT_{Capillary,Artery,Vein,LargeVessel}_seed42_test` |
| 大血管匹配对照 | `octa_ugr_fusion/runs/{octa,early}_GT_LargeVessel_seed42` |
| 大血管匹配对照测试 | `octa_ugr_fusion/runs/{octa,early}_GT_LargeVessel_seed42_test` |

目录中的花括号是简写。smoke_capillary、smoke_vein_topology、smoke_large_vessel 及 smoke_val_ablation* 仅为工程验证，不纳入正式性能排名。

现阶段最合适的研究表述是：**UGR 在 seed42 的三类历史测试比较中均取得最高 Dice，动脉提升最明显；但正确配对 OCT 相对打乱 OCT 的优势极小，尚无充分证据证明提升来自有效的多模态信息利用。** 稳定结论仍需同入口对照、容量与模块消融、多 seed 以及新的独立验证。
