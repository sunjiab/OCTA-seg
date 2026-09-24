# OCTA、OCT 与 Early Fusion 三组对照实验总结

> 更新日期：2026-09-24  
> 数据集：OCTA-500，3 mm 子集  
> 投影范围：ILM-OPL  
> 随机种子：42  
> 任务：GT_Capillary、GT_Artery、GT_Vein 三个独立二分类任务

## 1. 实验目的

本阶段比较三种输入方式：

| 方法 | 输入 | 目的 |
|---|---|---|
| OCTA-only | OCTA 投影 | 主要单模态基线 |
| OCT-only | 结构 OCT 投影 | 判断 OCT 单独包含多少血管信息 |
| Early Fusion | OCTA + OCT | 判断 OCT 是否能为 OCTA 分割提供增益 |

Early Fusion 在输入端沿通道维拼接：

```text
OCTA [B,1,304,304] ─┐
                    ├→ concat [B,2,304,304] → Shared U-Net → vessel mask
OCT  [B,1,304,304] ─┘
```

![OCTA + OCT Early Fusion](assets/octa_oct_early_fusion.png)

三个标签分别训练独立的二分类模型，不是一个联合区分毛细血管、动脉和静脉的多分类模型。

## 2. 共同实验设置

| 项目 | 设置 |
|---|---|
| Train | 10301-10440，共 140 例 |
| Validation | 10441-10450，共 10 例 |
| Test | 10451-10500，共 50 例 |
| 网络 | 四层 U-Net，base channels=32 |
| Loss | BCEWithLogits + Soft Dice，权重 1:1 |
| 优化器 | AdamW |
| 初始学习率 | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 8 |
| 最大 epoch | 100 |
| Early stopping | 验证 Dice 连续 20 轮不提升 |
| LR scheduler | ReduceLROnPlateau，factor=0.5，patience=5 |
| 二值化阈值 | 0.5 |
| 数据增强 | 同步 90° 旋转、水平翻转、垂直翻转 |
| 模型选择 | 验证集 micro Dice 最大 |
| 测试指标 | micro Dice、IoU、Precision、Recall、Specificity |

OCTA-only 使用原单模态 baseline 的正式结果；OCT-only 和 Early Fusion 使用本目录的训练入口。数据划分、U-Net 主体、Loss 和主要训练参数一致。Early Fusion 只将第一个卷积的输入通道从 1 改为 2，比单模态增加 288 个卷积权重。

## 3. 正式测试集结果

| 标签 | 输入 | Best epoch | Test loss | Dice | IoU | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GT_Capillary | OCTA | 57 | 0.3459 | 0.8953 | 0.8104 | 0.8728 | **0.9190** | 0.9043 |
| GT_Capillary | OCT | 3 | 1.1703 | 0.6207 | 0.4500 | 0.4759 | 0.8919 | 0.2984 |
| GT_Capillary | OCTA+OCT | 55 | 0.3446 | **0.8954** | **0.8107** | **0.8782** | 0.9133 | **0.9095** |
| GT_Artery | OCTA | 70 | 0.1425 | 0.9008 | 0.8194 | 0.8947 | **0.9069** | 0.9958 |
| GT_Artery | OCT | 97 | 0.5227 | 0.6373 | 0.4677 | 0.7257 | 0.5681 | 0.9915 |
| GT_Artery | OCTA+OCT | 78 | 0.1403 | **0.9019** | **0.8213** | **0.9071** | 0.8967 | **0.9964** |
| GT_Vein | OCTA | 81 | 0.1586 | 0.8810 | 0.7873 | 0.8895 | **0.8727** | 0.9968 |
| GT_Vein | OCT | 1 | 1.5466 | 0.0183 | 0.0092 | 0.0348 | 0.0124 | 0.9897 |
| GT_Vein | OCTA+OCT | 88 | 0.1554 | **0.8817** | **0.7884** | **0.8943** | 0.8695 | **0.9969** |

## 4. 最佳验证结果与训练停止情况

| 标签 | 输入 | 最佳轮次 | 实际训练轮数 | Best Val Dice | Best Val IoU |
|---|---|---:|---:|---:|---:|
| GT_Capillary | OCTA | 57 | 77 | 0.9028 | 0.8228 |
| GT_Capillary | OCT | 3 | 23 | 0.6400 | 0.4706 |
| GT_Capillary | OCTA+OCT | 55 | 75 | 0.9023 | 0.8221 |
| GT_Artery | OCTA | 70 | 90 | 0.8979 | 0.8148 |
| GT_Artery | OCT | 97 | 100 | 0.6599 | 0.4924 |
| GT_Artery | OCTA+OCT | 78 | 98 | 0.9005 | 0.8191 |
| GT_Vein | OCTA | 81 | 100 | 0.8883 | 0.7990 |
| GT_Vein | OCT | 1 | 21 | 0.0175 | 0.0088 |
| GT_Vein | OCTA+OCT | 88 | 100 | 0.8823 | 0.7895 |

OCT-only 毛细血管和静脉较早停止。毛细血管从第 3 轮后没有刷新验证 Dice；静脉从第 1 轮后没有刷新，并逐渐变成全背景预测。

## 5. Fusion 相对 OCTA-only 的变化

| 标签 | Dice | IoU | Precision | Recall | Specificity | Loss |
|---|---:|---:|---:|---:|---:|---:|
| GT_Capillary | +0.0001 | +0.0002 | +0.0055 | -0.0057 | +0.0052 | -0.0013 |
| GT_Artery | +0.0011 | +0.0018 | +0.0124 | -0.0103 | +0.0006 | -0.0022 |
| GT_Vein | +0.0007 | +0.0011 | +0.0048 | -0.0032 | +0.0002 | -0.0032 |

三个任务表现出相同方向：加入 OCT 后 Precision 和 Specificity 上升，Recall 下降。模型预测变得更保守，减少了一部分假阳性，也增加了少量漏检。

动脉的 Dice 增益最大，但也只有 0.0011，即约 0.11 个百分点。毛细血管和静脉的增益更小。

## 6. 验证集与测试集是否一致

| 标签 | OCTA Val Dice | Fusion Val Dice | Val 变化 | Test Dice 变化 |
|---|---:|---:|---:|---:|
| GT_Capillary | 0.9028 | 0.9023 | -0.0004 | +0.0001 |
| GT_Artery | 0.8979 | 0.9005 | +0.0026 | +0.0011 |
| GT_Vein | 0.8883 | 0.8823 | -0.0060 | +0.0007 |

融合在三个测试任务上都略高，但验证集只有动脉提高。毛细血管与静脉在验证集上下降，测试集却出现极小提升，说明当前结果存在抽样波动，不能仅根据测试集小数点后三位的变化认定融合有效。

当前只有一个 seed，验证集只有 10 例。0.0001-0.0011 的 Dice 差异很可能落在训练随机性范围内。现阶段合理结论是：

> Early Fusion 达到了与 OCTA-only 基本相同的性能，但尚未观察到稳定、显著的多模态增益。

## 7. OCT-only 结果分析

### 7.1 GT_Capillary

测试 Dice 为 0.6207，Recall 为 0.8919，但 Precision 只有 0.4759，Specificity 只有 0.2984。模型找到了大量真实血管，同时把许多背景结构错误预测为血管。

结构 OCT 投影中存在与血管位置相关的组织结构或投影阴影，但仅凭这些信息无法可靠区分血管和其他纹理，因此出现明显过分割。

### 7.2 GT_Artery

测试 Dice 为 0.6373，Precision 为 0.7257，Recall 为 0.5681。OCT 对大血管位置具有一定预测能力，但漏检较多。

额外的 200 epoch 设置实际训练至第 127 轮，最佳轮次为 107，测试 Dice 为 0.6384。相对默认实验只增加约 0.0011，说明增加训练轮数没有解决主要瓶颈。

### 7.3 GT_Vein

测试 Dice 只有 0.0183。训练集静脉前景像素约占 2.90%，背景约占 97.10%，背景与静脉像素比例约为 33.5:1。

最后一个模型在验证集上的最大输出概率为 0.4793，低于固定阈值 0.5，因此预测前景像素数为 0，Dice、IoU、Precision 和 Recall 均为 0，Specificity 接近 1。这是典型的全背景塌缩。

结构 OCT 缺少直接血流信号，也难以单独区分动脉和静脉；在严重类别不平衡下，预测背景可以较容易降低 BCE。因此，OCT-only 静脉实验适合作为消融中的负对照。

## 8. 对当前融合策略的判断

Early Fusion 只在输入端拼接两种模态，之后使用共享编码器。测试结果几乎复现 OCTA-only，可能有两种解释：

1. OCT 投影提供的额外信息有限；
2. 模型主要依赖 OCTA，第一层之后基本忽略 OCT。

仅凭当前指标不能区分这两种情况。下一步应对已经训练好的融合模型进行推理消融：

```text
正常 OCTA + 正常 OCT
正常 OCTA + 全零 OCT
正常 OCTA + 随机打乱病例的 OCT
```

如果置零或打乱 OCT 后指标基本不变，说明融合模型没有有效利用 OCT；如果性能明显下降，说明 OCT 确实提供了额外信息。

当前只按病例编号和图像尺寸检查配对，仍应通过并排图、关键结构或定量方法进一步确认两种投影的空间配准。

## 9. 当前可以得出的结论

- OCTA 是三个任务的主要有效模态，测试 Dice 为 0.8810-0.9008。
- OCT-only 明显弱于 OCTA-only，尤其无法独立完成静脉分割。
- Early Fusion 在三个测试任务上均略高于 OCTA-only。
- 测试 Dice 增益只有 0.0001-0.0011，验证集变化也不一致。
- 融合主要提高 Precision、降低 Recall，表现为减少误检并略微增加漏检。
- 当前结果可以作为 Early Fusion baseline，但不足以证明 OCT 能稳定提升 OCTA 分割。

## 10. 推荐的下一步实验

1. 使用 seed 42、43、44 重复 OCTA-only 与 Fusion，报告均值和标准差。
2. 对融合 checkpoint 做 OCT 置零和 OCT 病例打乱消融，验证网络是否使用 OCT。
3. 统计同一测试病例上 Fusion 与 OCTA-only 的逐病例 Dice 差值，而不只比较全局 micro Dice。
4. 增加 clDice，检查血管连通性和细血管断裂。
5. 检查 OCT/OCTA 空间对应，并比较 OCT(FULL)、OCT(ILM_OPL)、OCT(OPL_BM)。
6. 如果 OCT 确实有信息，再尝试双编码器或多尺度特征融合。

在完成多 seed 和模态置零消融前，实验报告建议使用以下措辞：

> 简单输入级融合与 OCTA-only 性能相当，在单次实验中获得轻微测试集提升，但尚未证明该提升具有稳定性。

## 11. 结果文件位置

OCTA-only：

```text
../octa_baseline/runs/unet_3mm_ilm_opl
../octa_baseline/runs/unet_3mm_artery
../octa_baseline/runs/unet_3mm_vein
```

OCT-only：

```text
runs/oct_3mm_ILM_OPL_GT_Capillary_seed42
runs/oct_3mm_ILM_OPL_GT_Artery_seed42
runs/oct_3mm_ILM_OPL_GT_Vein_seed42
```

Early Fusion：

```text
runs/fusion_3mm_ILM_OPL_GT_Capillary_seed42
runs/fusion_3mm_ILM_OPL_GT_Artery_seed42
runs/fusion_3mm_ILM_OPL_GT_Vein_seed42
```

每个正式实验目录包含配置、逐轮历史、最佳/最后 checkpoint、测试指标和预测图。统一可视化入口为 `结果与标签可视化.ipynb`。
