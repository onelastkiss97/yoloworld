# YOLO-World 开放词表与 UAV 5-shot 实验交接

更新日期：2026-08-24  
原实验目录：`/home/songziqi/project/YOLO-World/openworld_1`  
代码仓库：`/home/songziqi/project/YOLO-World`  
Conda 环境：`YOLO-World`  

## 1. 最重要的当前结论

本项目已经完成三部分实验：

1. 在 `multirotor drone`、`helicopter` 两个 base 类上进行 YOLO-World V2-M 文本条件微调，并以未参与训练的 `fixed-wing aircraft` 作为 novel 类进行开放词表测试。
2. 使用 base 微调所得的 epoch 100 **未重参数化、仍可输入文本 prompt 的 checkpoint**，以 `small drone` 为 prompt 在 UAV/RGB test 上完成 zero-shot 测试。
3. 使用 UAV/RGB train 中 5 张图，加上 openworld_base 的 15 张 replay 图进行 5-shot 微调；模型选择只看 UAV/RGB val，最后只在 UAV/RGB test 上评估一次。

容易混淆但必须明确：

- 5-shot 的训练数据确实是两个来源的混合数据（5 张 UAV/RGB + 15 张 openworld_base replay）。
- 最终报告的 `mAP50=0.510` **仅在 UAV/RGB 完整 test 集上计算**，没有把 openworld_base 混入验证或测试。
- UAV/RGB test 有 614 张图片、552 个框，只按一个评估类别 `small drone` 计算指标。
- zero-shot 和 5-shot 使用的是同一个 UAV/RGB test，因此 `0.451 -> 0.510` 可以直接比较。

## 2. Base-to-novel 开放词表实验

### 任务设定

- base 训练类：`multirotor drone`、`helicopter`
- novel 测试类：`fixed-wing aircraft`
- 原始预训练权重：
  `/home/songziqi/project/YOLO-World/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth`
- 框架：官方 MMYOLO / YOLO-World，而不是 Ultralytics
- 输入分辨率：1280
- 训练：150 epochs，batch 4，AdamW，LR `2e-4`
- 每 5 epochs 在两类 `val_base` 上验证，以 COCO bbox mAP 选择最佳 checkpoint
- 最后 10 epochs 关闭 Mosaic

### 数据组织

数据根目录：

`/home/songziqi/project/Data/openworld_base`

原始 train/val/test 得到保留；额外构建了无 fixed-wing 泄漏的两类训练/验证视图：

- `train_base`：6269 张，6625 个框；排除 443 张含 fixed-wing 的训练图
- `val_base`：1461 张，1643 个框；排除 55 张含 fixed-wing 的验证图
- 完整 test：938 张，997 个框，其中 fixed-wing 56 个框

COCO 标注：

- `/home/songziqi/project/Data/openworld_base/annotations/instances_train_base.json`
- `/home/songziqi/project/Data/openworld_base/annotations/instances_val_base.json`
- `/home/songziqi/project/Data/openworld_base/annotations/instances_test_openworld.json`

YAML 与审计：

- `/home/songziqi/project/Data/openworld_base/data_base.yaml`
- `/home/songziqi/project/Data/openworld_base/data_openworld_test.yaml`
- `/home/songziqi/project/Data/openworld_base/openworld_ovd_audit.json`
- 数据重建/校验脚本：`/home/songziqi/project/YOLO-World/openworld_1/prepare_openworld_ovd.py`

类别顺序固定为：

- 0 = `multirotor drone`
- 1 = `helicopter`
- 2 = `fixed-wing aircraft`

`fixed-wing aircraft` 没有参与 base 微调。它只在最终测试/导出阶段作为新增文本提示生成文本特征，然后与最佳两类 checkpoint 融合并重参数化。

### 关键代码修复

早期错误来自继承 mask-refine 训练基配置。UAV 数据只有 bbox、没有 COCO segmentation mask；mask-refine 增强流水线会把 bbox 清空，导致 bbox/DFL loss 恒为 0。

最终改为纯检测基配置：

`third_party/mmyolo/configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py`

并使用标准 MMDetection 数据流：

- `DetDataPreprocessor`
- `list[DetDataSample]`
- `InstanceData`

自定义组件：

`/home/songziqi/project/YOLO-World/openworld_1/standard_ovd_components.py`

其中：

- `StandardYOLOWorldHead` 处理 `txt_masks + InstanceData`
- `PositiveDetectionLossHook` 检查训练初期 bbox/DFL loss 非零，防止再次静默训练空框
- `FrozenBackboneYOLOWorldDetector` 用于后续 5-shot，在优化器创建前冻结图像与文本 backbone

训练入口包装：

- `/home/songziqi/project/YOLO-World/openworld_1/train_with_registered_mmyolo.py`
- `/home/songziqi/project/YOLO-World/openworld_1/test_with_registered_mmyolo.py`

### 关键文件与结果

训练配置：

`/home/songziqi/project/YOLO-World/openworld_1/yolo_world_v2_m_openworld_standard_1280.py`

Slurm 脚本：

`/home/songziqi/project/YOLO-World/openworld_1/yoloworld_openworld_standard_train.sh`

Slurm job：`78291`，已完成。完整日志：

`/home/songziqi/Slurmout/yw_ovd_standard-78291.out`

最佳文本条件 checkpoint（epoch 100）：

`/home/songziqi/project/YOLO-World/openworld_1/runs/v2_m_ovd_standard_base2_novel_fixedwing_1280_e150/best_coco_bbox_mAP_epoch_100.pth`

此 checkpoint 仍包含文本分支，后续要更换 prompt、做 zero-shot 或继续微调时应优先使用它。

最终三类固定词表重参数化权重：

`/home/songziqi/project/YOLO-World/openworld_1/weights/v2_m_ovd_standard_base2_novel_fixedwing_1280_e150/best_coco_bbox_mAP_epoch_100_rep_conv.pth`

三类完整 test 结果：

| 类别 | AP | AP50 | AP75 |
|---|---:|---:|---:|
| multirotor drone | 0.611 | 0.976 | 0.669 |
| helicopter | 0.629 | 0.931 | 0.652 |
| fixed-wing aircraft（novel） | 0.179 | 0.395 | 0.186 |
| 总体 | 0.473 | 0.767 | 0.502 |

重参数化只是在训练结束后把指定文本特征固化到检测结构中，不会让 fixed-wing 获得新的图像监督。若直接使用 epoch 100 文本条件 checkpoint 并输入相同三类 prompt，理论检测能力来源相同；重参数化主要用于固定词表部署和推理效率。

## 3. UAV/RGB zero-shot

数据根目录：

`/home/songziqi/project/Data/UAV/RGB`

prompt 必须是：

`small drone`

zero-shot 使用上述 epoch 100 文本条件 checkpoint，没有在 UAV/RGB 上训练。

配置与 Slurm：

- `/home/songziqi/project/YOLO-World/openworld_1/0-shots/yolo_world_v2_m_uav_rgb_zeroshot_1280.py`
- `/home/songziqi/project/YOLO-World/openworld_1/0-shots/eval_uav_rgb_best100_small_drone.sh`

Slurm job：`78529`，已完成。日志：

`/home/songziqi/project/YOLO-World/openworld_1/0-shots/slurm_logs/yw_ovd_uav_zs-78529.out`

UAV/RGB test zero-shot 指标：

| mAP | mAP50 | mAP75 | AP-small | AP-medium | AP-large |
|---:|---:|---:|---:|---:|---:|
| 0.186 | 0.451 | 0.125 | 0.042 | 0.330 | 0.343 |

预测结果：

`/home/songziqi/project/YOLO-World/openworld_1/0-shots/uav_rgb_best_epoch100_prompt_small_drone/predictions.pkl`

日志的 COCO 类别表曾显示 `uav`，这是评估标注 metainfo 的显示名称，不代表模型 prompt。实际传入模型的 prompt 已核验为 `small drone`。

## 4. UAV/RGB 5-shot + base replay

### 训练数据

5 张 UAV/RGB 图片均从原始 train 中选择，未与 val/test 重叠：

- `01001.png`
- `05150.png`
- `36020.png`
- `38108.png`
- `49020.png`

源位置：

- 图片：`/home/songziqi/project/Data/UAV/5-shots/images/train`
- 标签：`/home/songziqi/project/Data/UAV/5-shots/labels/train`

`labels/train` 仍然有用：它是重建 mixed_train 的源标签。实际训练读取的是 `mixed_train/labels`。

为减少 base 类遗忘，训练中还加入 15 张从 openworld_base/train_base 确定性挑选的 replay 图：

- 8 张 multirotor 图
- 7 张 helicopter 图

最终混合训练集：

- 20 张图
- 23 个框
- `small drone`：5 个框
- `multirotor drone`：8 个框
- `helicopter`：10 个框

训练类别顺序：

- 0 = `small drone`
- 1 = `multirotor drone`
- 2 = `helicopter`

关键数据文件：

- 构建脚本：`/home/songziqi/project/Data/UAV/5-shots/prepare_5shot_replay.py`
- 实际训练视图：`/home/songziqi/project/Data/UAV/5-shots/mixed_train`
- 训练 COCO：`/home/songziqi/project/Data/UAV/5-shots/annotations/instances_train_5shot_replay.json`
- val COCO：`/home/songziqi/project/Data/UAV/5-shots/annotations/instances_val_small_drone.json`
- test COCO：`/home/songziqi/project/Data/UAV/5-shots/annotations/instances_test_small_drone.json`
- 训练 prompt：`/home/songziqi/project/Data/UAV/5-shots/texts/train_class_texts.json`
- 评估 prompt：`/home/songziqi/project/Data/UAV/5-shots/texts/eval_class_texts.json`
- 完整审计：`/home/songziqi/project/Data/UAV/5-shots/five_shot_replay_audit.json`

注意：val/test 的图片仍从 `/home/songziqi/project/Data/UAV/RGB/images/val` 和 `images/test` 读取。它们不包含 replay 数据，并且只评估 `small drone`。

### 微调设置

- 起点：epoch 100 的未重参数化 checkpoint
- prompt：UAV 为 `small drone`，replay 为 `multirotor drone` / `helicopter`
- 冻结完整 CLIP 文本 backbone
- 冻结完整图像 backbone
- 只训练 `neck.*` 与 `bbox_head.*`
- 审计通过：16,520,286 / 91,804,238 参数可训练，占 18.00%
- AdamW，恒定 LR `1e-5`
- 1000 iterations，batch 4
- 每 100 iterations 保存并在 UAV/RGB val 验证
- 以 `coco/bbox_mAP_50` 选最佳权重
- 1280 分辨率
- 关闭 Mosaic、MixUp、CopyPaste
- 保留水平翻转、HSV、0.9-1.1 缩放、0.05 平移
- 最终 test 只评估一次

配置与 Slurm：

- `/home/songziqi/project/YOLO-World/openworld_1/5-shots/yolo_world_v2_m_uav_5shot_replay_1000iter.py`
- `/home/songziqi/project/YOLO-World/openworld_1/5-shots/train_uav_5shot_replay_1000iter.sh`

Slurm job：`78614`，状态 `COMPLETED`，exit code 0，耗时约 14 分 20 秒。

完整日志：

`/home/songziqi/project/YOLO-World/openworld_1/5-shots/slurm_logs/yw_uav_5shot-78614.out`

### 验证曲线与最终测试

UAV/RGB val mAP50：

| iteration | mAP50 |
|---:|---:|
| 100 | 0.506 |
| 200 | **0.522** |
| 300 | 0.519 |
| 400 | 0.509 |
| 500 | 0.505 |
| 600 | 0.506 |
| 700 | 0.505 |
| 800 | 0.499 |
| 900 | 0.495 |
| 1000 | 0.492 |

最佳点为 iter 200，之后出现明显 few-shot 过拟合。

最佳 checkpoint：

`/home/songziqi/project/YOLO-World/openworld_1/5-shots/runs/v2_m_uav_5shot_replay_1000iter/best_coco_bbox_mAP_50_iter_200.pth`

在 UAV/RGB 完整 test（614 张、552 框）上的最终指标：

| mAP | mAP50 | mAP75 | AP-small | AP-medium | AP-large |
|---:|---:|---:|---:|---:|---:|
| 0.217 | 0.510 | 0.148 | 0.048 | 0.368 | 0.422 |

相对 zero-shot：

- mAP50：`0.451 -> 0.510`
- 绝对提升：`+0.059`
- 相对提升：约 `+13.1%`

预测结果：

`/home/songziqi/project/YOLO-World/openworld_1/5-shots/test_results/v2_m_uav_5shot_replay_1000iter_best_map50/predictions.pkl`

## 5. Slurm 环境约定

已使用的资源约定：

- partition：`gpu4090`
- GPU：1
- CPU：8
- memory：32 GB
- `#SBATCH --time=0`，表示请求无显式 walltime；仍受分区/QoS 上限约束
- 训练完成后脚本自然退出
- Conda：`source /home/songziqi/anaconda3/etc/profile.d/conda.sh && conda activate YOLO-World`
- Python：`/home/songziqi/anaconda3/envs/YOLO-World/bin/python`
- 离线 CLIP：`/home/songziqi/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`

提交示例：

```bash
sbatch /home/songziqi/project/YOLO-World/openworld_1/5-shots/train_uav_5shot_replay_1000iter.sh
```

不要直接重复提交上述 5-shot 脚本来做对比实验，因为它会复用相同 work_dir。新实验应先复制配置/脚本并更换 run tag、work_dir 和 test output 目录。

## 6. 推荐后续工作

当前最大瓶颈是小目标：AP-small 只从 `0.042` 提高到 `0.048`。若继续以提升 UAV/RGB mAP50 为目标，优先顺序建议：

1. 围绕 100-300 iterations 做更密集验证（每 25 或 50 iter），因为当前最佳已在 iter 200。
2. 比较 replay 与 no-replay；两者的最终 UAV 指标必须都只在同一个 UAV/RGB test 上计算。
3. 尝试 LR `5e-6` 或只部分解冻 neck，降低 5 张样本上的过拟合。
4. 针对小目标尝试 crop/tiling 或小目标过采样；必须注意验证与测试推理方式保持一致。
5. 用多个独立 5-shot 采样重复实验，报告均值和标准差，避免单次样本选择偶然性。
6. 如需部署，可测试把 5-shot 最佳 checkpoint 用 `small drone` 词表重参数化后，mAP50 是否保持一致。

## 7. 给新窗口的最短读取指令

新窗口开始时可直接说：

> 请先完整阅读 `/home/songziqi/project/YOLO-World/newwork/openworld_1/HANDOFF.md`，然后基于其中现有文件继续工作。不要重建已完成的数据集，不要覆盖已有 work_dir。当前重点是提高 UAV/RGB test 的 small-drone mAP50；现有 zero-shot 为 0.451，5-shot 最佳 test 为 0.510，训练混合了 15 张 base replay，但 val/test 都只使用 UAV/RGB。

