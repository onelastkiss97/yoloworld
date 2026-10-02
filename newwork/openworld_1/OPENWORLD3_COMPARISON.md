# Sparse-Zoom 第三类 zero-shot 测试对比

测试日期：2026-08-25  
Slurm 作业：`80099`

## 测试口径

- 数据：`openworld_base` 完整 test，938 张图片、997 个框
- 标注：`annotations/instances_test_openworld.json`
- 类别顺序：`multirotor drone`、`helicopter`、`fixed-wing aircraft`
- 文本提示：`a multirotor drone`、`a helicopter`、`a fixed-wing aircraft`
- base 训练类：前两类
- novel 测试类：`fixed-wing aircraft`，训练中未使用该类图像或标注
- 输入分辨率：1280
- 新模型保持动态文本分支，没有进行固定词表重参数化

新模型权重：

`runs/sparse_zoom_v2_m_ovd_base2_1280_e80/best_coco_bbox_mAP_epoch_80.pth`

原结果来自 `HANDOFF.md` 中 epoch 100 标准 YOLO-World 的三类固定词表测试。

## 类别指标

| 类别 | 指标 | 原模型 | Sparse-Zoom | 绝对变化 |
|---|---:|---:|---:|---:|
| multirotor drone | AP | 0.611 | 0.600 | -0.011 |
|  | AP50 | 0.976 | 0.966 | -0.010 |
|  | AP75 | 0.669 | 0.676 | +0.007 |
| helicopter | AP | 0.629 | 0.596 | -0.033 |
|  | AP50 | 0.931 | 0.918 | -0.013 |
|  | AP75 | 0.652 | 0.589 | -0.063 |
| fixed-wing aircraft (novel) | AP | 0.179 | 0.063 | -0.116 |
|  | AP50 | 0.395 | 0.147 | -0.248 |
|  | AP75 | 0.186 | 0.044 | -0.142 |

第三类相对变化：AP 下降约 64.8%，AP50 下降约 62.8%，AP75 下降约 76.3%。

## 总体指标

| 指标 | 原模型 | Sparse-Zoom | 绝对变化 |
|---|---:|---:|---:|
| AP | 0.473 | 0.419 | -0.054 |
| AP50 | 0.767 | 0.677 | -0.090 |
| AP75 | 0.502 | 0.436 | -0.066 |
| AP-small | 0.416 | 0.432 | +0.016 |
| AP-medium | 0.452 | 0.358 | -0.094 |
| AP-large | 0.495 | 0.441 | -0.054 |

## 结论

Sparse-Zoom 在相同 test 上带来轻微的小目标总体收益（AP-small `+0.016`），但第三类 zero-shot 泛化显著退化。两类 base 的平均 AP 从 0.620 降至 0.598（`-0.022`），而 novel AP 下降 `-0.116`，因此总体下降主要来自 novel 类，而不是单纯的 base 检测能力下降。

当前结果不支持“稀疏高分辨率模块提升开放词表 novel 类检测”的结论。下一步应优先做开放词表保持策略，例如冻结原 P3--P5 文本条件分类路径、对新 P2/router 使用残差门控，以及在不引入 fixed-wing 图像监督的前提下加入通用词表蒸馏。后续实验仍须保持 fixed-wing 不参与训练。

## 产物

- 配置：`sparse_zoom_ovd_openworld3_test_1280.py`
- Slurm：`eval_sparse_zoom_openworld3_best80.sh`
- 日志：`slurm_logs/yw_sp_ovd3-80099.out`
- 预测：`test_results/sparse_zoom_best80_openworld3_dynamic/predictions.pkl`

