# OCTA-500 单模态血管分割基线

这是研究路线第一阶段的可复现实验实现：

```text
OCTA(ILM_OPL) -> U-Net -> binary vessel mask
```

默认设置为 3 mm 子集、`GT_Capillary` 二值血管标签、BCE + Dice loss，并统计 Dice、IoU、Precision、Recall 和 Specificity。

## 数据划分

按病例编号划分，避免病例泄漏：

| 子集 | Train | Validation | Test |
|---|---|---|---|
| 3 mm | 10301-10440 (140) | 10441-10450 (10) | 10451-10500 (50) |
| 6 mm | 10001-10180 (180) | 10181-10200 (20) | 10201-10300 (100) |

3 mm 和 6 mm 分辨率不同，因此默认分开训练，不在同一个 batch 中混用。

## 环境

环境已经创建后可直接激活：

```bash
conda activate octa
```

若需在另一台机器重建：

```bash
conda env create -f octa_baseline/environment.yml
conda run -n octa python -m pip install --no-deps -e octa_baseline
```

## 快速验证

在项目根目录运行：

```bash
conda run -n octa pytest -q octa_baseline/tests
```

## 训练默认 3 mm baseline

```bash
cd octa_baseline
conda run -n octa python train.py \
  --data-root ../data/OCTA-500数据集 \
  --scan-size 3mm \
  --projection 'OCTA(ILM_OPL)' \
  --target GT_Capillary \
  --epochs 100 \
  --batch-size 8 \
  --output-dir runs/unet_3mm_ilm_opl
```

训练结束后输出目录包含：

- `config.json`：完整实验参数
- `history.csv`：逐 epoch 训练/验证指标
- `tensorboard/`：TensorBoard 日志
- `best.pt`：验证集 Dice 最优模型
- `last.pt`：最后一个 epoch 模型，支持恢复训练
- `test_metrics.json`：最优模型的测试集指标
- `test_predictions.png`：测试集输入、真值、概率图和预测图

查看训练曲线：

```bash
conda run -n octa tensorboard --logdir octa_baseline/runs
```

## 结果与标签可视化

训练完成后打开 `octa_baseline/结果与标签可视化.ipynb`，选择 `Python (octa)` 内核并运行全部单元格。Notebook 会展示测试集总体指标，以及输入、Ground Truth、概率图、二值预测、TP/FP/FN 误差图和轮廓叠加图。若使用了不同的输出目录，只需修改 Notebook 顶部的 `RUN_DIR`。

## 其他单模态对照

只需修改参数即可复用完全相同的流程：

```bash
# 6 mm baseline
conda run -n octa python octa_baseline/train.py \
  --data-root data/OCTA-500数据集 --scan-size 6mm \
  --output-dir octa_baseline/runs/unet_6mm_ilm_opl

# FULL projection ablation
conda run -n octa python octa_baseline/train.py \
  --data-root data/OCTA-500数据集 --scan-size 3mm \
  --projection 'OCTA(FULL)' \
  --output-dir octa_baseline/runs/unet_3mm_full
```

不要把同一个模型在验证阶段反复选择阈值后再报告该验证结果。当前实现固定使用阈值 0.5，并只在验证 Dice 最优时保存模型，最后只评估一次测试集。
