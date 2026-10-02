# Sparse-Zoom UAV/RGB 5-shot 对比

测试日期：2026-08-25  
Slurm 作业：`80102`

## 实验口径

本实验严格复用 `HANDOFF.md` 的数据边界和模型选择方式：

- 训练：5 张 UAV/RGB train 图片 + 15 张 openworld base replay 图片
- 混合训练集：20 张图、23 个框
- 类别顺序：`small drone`、`multirotor drone`、`helicopter`
- 验证：只使用 UAV/RGB val（616 张、531 个框）
- 测试：只使用 UAV/RGB test（614 张、552 个框）
- 输入：1280
- 优化：AdamW，恒定 LR `1e-5`，batch 4，1000 iterations
- 每 100 iterations 验证，以 UAV val `bbox_mAP_50` 选择最佳权重
- 冻结完整 CLIP 和图像 backbone
- 仅训练 Sparse-Zoom neck/router 与 P2--P5 bbox head
- 可训练参数：17,432,962 / 92,716,914（18.80%）
- 起始权重：openworld base2 的动态 `best_coco_bbox_mAP_epoch_80.pth`

原始数据集没有被重建或覆盖。

## 验证曲线

| iteration | 原标准模型 mAP50 | Sparse-Zoom mAP50 |
|---:|---:|---:|
| 100 | 0.506 | 0.457 |
| 200 | **0.522** | **0.474** |
| 300 | 0.519 | 0.467 |
| 400 | 0.509 | 0.466 |
| 500 | 0.505 | 0.472 |
| 600 | 0.506 | 0.461 |
| 700 | 0.505 | 0.461 |
| 800 | 0.499 | 0.461 |
| 900 | 0.495 | 0.459 |
| 1000 | 0.492 | 0.462 |

两种模型均在 iter 200 达到最佳 val mAP50，之后发生 few-shot 过拟合。

## UAV/RGB test 结果

| 模型 | mAP | mAP50 | mAP75 | AP-small | AP-medium | AP-large |
|---|---:|---:|---:|---:|---:|---:|
| 原标准模型 zero-shot | 0.186 | 0.451 | 0.125 | 0.042 | 0.330 | 0.343 |
| 原标准模型 5-shot | 0.217 | **0.510** | 0.148 | **0.048** | **0.368** | 0.422 |
| Sparse-Zoom zero-shot | 0.146 | 0.378 | 0.092 | 0.030 | 0.275 | 0.339 |
| Sparse-Zoom 5-shot | 0.212 | 0.463 | **0.179** | 0.030 | 0.361 | **0.446** |

Sparse-Zoom 自身从 zero-shot 到 5-shot 的绝对变化：

- mAP：`0.146 -> 0.212`（`+0.066`）
- mAP50：`0.378 -> 0.463`（`+0.085`，相对提升约 22.5%）
- mAP75：`0.092 -> 0.179`（`+0.087`）
- AP-small：`0.030 -> 0.030`（无提升）
- AP-medium：`0.275 -> 0.361`（`+0.086`）
- AP-large：`0.339 -> 0.446`（`+0.107`）

Sparse-Zoom 5-shot 相对原标准 5-shot：

- mAP：`-0.005`
- mAP50：`-0.047`
- mAP75：`+0.031`
- AP-small：`-0.018`
- AP-medium：`-0.007`
- AP-large：`+0.024`

## 结论和诊断

本次 5-shot 确实显著改善了 Sparse-Zoom 自身的 UAV 迁移能力，而且 AP75
和 AP-large 超过原标准 5-shot；但目标指标 mAP50 和 AP-small 仍落后，因此
当前实验不支持 Sparse-Zoom 提升 UAV 小目标检测的结论。

关键原因是当前 5-shot 采样与路由器的小目标定义不匹配。路由器只对 1280
输入中几何尺寸不超过 96 像素的框计算 coverage 监督，而 5 张 UAV 样本中的
框尺寸约为 232.0、387.9、427.0、389.0、250.5 像素，全部超过阈值。整个
混合训练集 23 个框中只有 2 个满足该阈值，因此训练日志中的
`loss_router_coverage` 几乎一直为 0。模型主要学到了常规 neck/head 适配，P2
稀疏路由没有获得足够的小目标正监督。

下一步应该固定其余协议，只更换 5 张 UAV 样本为真正的小目标样本，并报告
多个独立 5-shot split 的均值和标准差；或者提高/自适应 router 的小目标阈值。
任何新 split 仍必须只从 UAV train 选择，不能与 val/test 重叠。

## 产物

- 配置：`sparse_zoom_uav_5shot_replay_1000iter.py`
- Slurm：`train_sparse_zoom_uav_5shot_replay.sh`
- 日志：`5-shots/slurm_logs/yw_sp_uav5-80102.out`
- 最佳权重：`5-shots/runs/sparse_zoom_uav_5shot_replay_1000iter/best_coco_bbox_mAP_50_iter_200.pth`
- zero-shot 预测：`5-shots/zero_shot/sparse_zoom_best80_small_drone/predictions.pkl`
- 5-shot 预测：`5-shots/test_results/sparse_zoom_uav_5shot_replay_best_map50/predictions.pkl`

