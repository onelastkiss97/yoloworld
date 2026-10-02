# YOLO-World：从公开预训练、少样本适配到低延迟部署的完整实验总结

整理日期：2026-09-12。本文依据当前保留的配置、代码、样本清单、训练日志、测试 JSON 和测速 JSON 编写。本次整理只新增本文，没有重新训练或修改模型。

路径约定：下文以 `openworld_1/`、`openworld_2/`、`openworld_3/` 开头的路径，均相对于 `/home/songziqi/project/YOLO-World/`；可点击链接以本文所在目录为起点。

## 1. 最终结论和正确的实验路线

最终实现的是：**在同一个 5-shot 模型上，fixed-wing aircraft 测试 AP50 保持 79.89%，RTX 3090 上平均纯推理降到 6.02 ms。** 输入保持 1280×1280，batch=1，三个类别不变，没有通过缩小图片或增加训练样本实现加速。

需要把两类工作分开理解：

- **提高新类别精度：**引入少量 fixed-wing 标注，冻结编码器，训练原有 neck/head，并回放旧类别样本。
- **降低推理耗时：**固定文本、FP16、算子重参数化、TensorRT 和 CUDA Graph。这一阶段没有重新训练，其目标是尽量保持精度，而不是继续增加 AP。

真实的权重关系如下；10-shot 与 5-shot 是两个独立分支，不是累计学习：

```text
公开 YOLO-World V2-M 1280 预训练权重
    │ 本地只训练 multirotor drone / helicopter
    ▼
基础模型：base-only 训练，验证选中 epoch 100
    ├── 三类测试：fixed-wing 无本地训练标注，AP50=39.47%
    ├── 10-shot + 40 张 base replay → 验证选 iter 500 → 测试 AP50=82.99%
    └──  5-shot + 40 张 base replay → 验证选 iter 400 → 测试 AP50=80.14%
                                               │ 不再训练
                                               ▼
                                固定文本 / 分类头折叠 / Conv-BN 融合
                                               │ ONNX → TensorRT FP16
                                               ▼
                                 CUDA Graph：6.02 ms，AP50=79.89%
```

本项目中的“零样本”是指：**本地 base-only 训练阶段没有使用 fixed-wing 训练标注**，随后用文本提示检测该类别。它不等于证明公开预训练模型从未见过飞机，也不是其他目录中名称包含 `0-shots` 的实验。

### 1.1 精度阶段总表

下表为同一原始 938 张三类测试集；AP50 是单类别 AP@IoU=0.50，最后一列是三类 COCO mAP@0.50:0.95。

| 阶段 | 本地 fixed-wing 训练图像/实例 | fixed-wing AP50 | fixed-wing mAP | multirotor AP50 | helicopter AP50 | 三类总体 mAP |
|---|---:|---:|---:|---:|---:|---:|
| 公开预训练权重直接测试 | 尚未做本地适配 | 未保留独立实测 | 未保留独立实测 | 未保留独立实测 | 未保留独立实测 | 未保留独立实测 |
| base-only epoch100，基线复测 | 0 / 0 | 39.47% | 0.179 | 97.63% | 93.11% | 0.473 |
| 10-shot，iter500 | 10 / 10 | 82.99% | 0.348 | 95.29% | 95.51% | 0.491 |
| 5-shot，iter400，原 test 输出 | 5 / 5 | 80.14% | 0.310 | 94.44% | 95.85% | 0.463 |
| 同一 5-shot，缓存文本 FP32 复测 | 5 / 5，未新增 | 80.16% | 0.309645 | 94.44% | 95.85% | 约0.462614¹ |
| 同一 5-shot，PyTorch 分类头折叠 FP16 + Graph | 5 / 5，未新增 | 80.11% | 0.309515 | 95.28% | 95.84% | 约0.462521¹ |
| 同一 5-shot，最终 TensorRT FP16 + Graph | 5 / 5，未新增 | **79.89%** | **0.311302** | 95.26% | 95.85% | 约0.463117¹ |

¹ 根据相应 JSON 的三个单类 mAP 求算术平均，不是另一次测试。表中其他三位小数总体指标直接取原测试摘要。

**不能把这个过程写成“所有类别、所有 mAP 一直上升”。** 新类别 AP50 大幅提高，但 multirotor AP50 相比 base-only 下降；5-shot 的总体 mAP 也略低于原基线。回放旨在缓解遗忘，不保证完全保留旧类别性能。10-shot 比5-shot高约2.85个百分点，但这里只做了一个固定样本划分，不能据此推导所有抽样条件下的普遍结论。

证据：[基线复测 JSON](../openworld_1/fixedwing_adapt/runs/baseline_85511/20260912_154458/20260912_154458.json)、[10-shot 摘要](../openworld_1/fixedwing_adapt/test_results/neck_10shot_85512/summary.json)、[5-shot 摘要](../openworld_2/test_results/neck_5shot_85544/summary.json)、[最终加速精度](results/tensorrt_fp16_graph_accuracy.json)。

## 2. 数据长什么样，训练/验证/测试分别做什么

### 2.1 数据规模

原始数据位于 `/home/songziqi/project/Data/openworld_base`。审计读取原始图像和标注，实验数据清单转为 COCO JSON；原图片和原标注未因本轮优化被改写。

| 原始划分 | 总图像数 | fixed-wing 图像数 | fixed-wing 框数 | multirotor 框数 | helicopter 框数 |
|---|---:|---:|---:|---:|---:|
| train | 6,712 | 443 | 467 | 4,405 | 2,220 |
| val | 1,516 | 55 | 55 | 977 | 666 |
| test | 938 | 56 | 56 | 608 | 333 |

本地基础训练使用其中 **6,269 张 base-only 图像**，不使用443张含 fixed-wing 的图像。少样本阶段才从原 train 中选出5张或10张 fixed-wing 图像。

数据包含天空、跑道、航模等场景，飞机形状细长、尺度较大；审计接触图还发现部分标签语义可能存在歧义。查看 [train 接触图](../openworld_1/fixedwing_adapt/audit/train_fixedwing.jpg)、[val 接触图](../openworld_1/fixedwing_adapt/audit/val_fixedwing.jpg)、[test 接触图](../openworld_1/fixedwing_adapt/audit/test_fixedwing.jpg)。没有为了提高结果而修正测试标签。

### 2.2 三个划分的职责

- **train/support：**参与反向传播，真正更新权重。重复采样和增强不会增加独立 shot 数。
- **val：**不反向传播；每100个训练迭代评估一次，用 `coco/fixedwing_ap50` 选择最佳权重。用户已允许使用原 val 的额外55张 novel 标注。
- **test：**对验证集选好的权重报告结果，不用于选择微调迭代或重抽 support。加速阶段继续用相同 test 检查数值精度变化。

因此规范表述应为：**5-shot/10-shot 训练，加上55张额外 novel 验证标注用于选模**，而不是“整个实验只用过5/10张带标注的新类图片”。

### 2.3 数据审计与局限

[audit/report.json](../openworld_1/fixedwing_adapt/audit/report.json) 记录 **463组跨划分像素级重复图像**，涉及 base 类；未发现 fixed-wing 的跨划分像素级完全重复。另发现2个 novel 感知哈希近似候选。Support 选择排除与 held-out 完全重复或近似的 novel 图像，replay 排除与 held-out 完全重复的图像；5-shot 脚本再次检查像素哈希隔离。

这不证明整套数据完全无泄露：base 模型已受原划分重叠影响，同源视频的不同帧也可能相关。审计还读取 held-out 图像进行去重，因此实验并非完全盲于 held-out 的数据整理流程；但没有用 test AP 挑选训练图片。最终结果应解释为**在现有固定测试集上的少样本适配及部署对比**，不是全新清洁数据划分上的无偏泛化结论。

## 3. 阶段A：公开预训练模型 → 本地 base-only 基础模型

### 3.1 起点与原有模块

起点权重为项目根目录中的：

`yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth`

这是已经做过视觉—语言检测预训练的 YOLO-World V2-M，而不是随机初始化网络。其在本地三类数据上“完全不训练直接测试”的结果当前没有单独记录，不能把后面的39.47%说成原始公开权重的直接性能。

| 原有组成 | 对应代码/配置类 | 作用 |
|---|---|---|
| 图像编码器 | `MultiModalYOLOBackbone.image_model`，继承 YOLOv8-M backbone 配置 | 提取多尺度视觉特征 |
| 文本编码器 | `HuggingCLIPLanguageBackbone`，CLIP ViT-B/32 文本模型 | 把类别描述变成512维语义向量 |
| 文本引导 neck | `YOLOWorldPAFPN`、`MaxSigmoidCSPLayerWithTwoConv` | 融合多尺度图像特征，利用文本引导注意力 |
| 检测头 | `YOLOWorldHeadModule`，`use_bn_head=True` | 产生边框与视觉—文本类别匹配分数 |

核心源码：[检测器](../yolo_world/models/detectors/yolo_world.py)、[检测头](../yolo_world/models/dense_heads/yolo_world_head.py)、[注意力模块](../yolo_world/models/layers/yolo_bricks.py)。这些属于原项目，不是本轮新发明或新添加的检测架构。

### 3.2 基础训练做了什么

[基础配置](../openworld_1/yolo_world_v2_m_openworld_standard_1280.py) 指定：仅两个 base 类、1280输入、batch=4、AdamW学习率 `2e-4`、weight decay `0.05`、最多150 epoch；CLIP文本编码器冻结。采用 Mosaic/MixUp 等增强，最后10个epoch切换增强流程，并使用 EMA。此时不能套用后面少样本阶段“图像编码器也全部冻结”的说法。

基础验证按 `coco/bbox_mAP`（IoU 0.50:0.95）选权重，保留的最佳权重是：

`openworld_1/runs/v2_m_ovd_standard_base2_novel_fixedwing_1280_e150/best_coco_bbox_mAP_epoch_100.pth`

测试时扩展为三个类别文本，fixed-wing 没有在本地基础训练中提供监督。历史表格显示 AP50=0.395；保留的原多模态模型复测精确值为 **0.3947218898**，与历史四舍五入结果一致。基础训练部分旧日志已按先前清理要求移除，因此不补造每个 epoch 的提升曲线。

### 3.3 本阶段文件与功能

以下是这一实验层新增/保留的配置与适配代码，不是本文整理时新建的文件。

| 文件 | 对应处理手段 | 是否是“提点模块” |
|---|---|---|
| `openworld_1/yolo_world_v2_m_openworld_standard_1280.py` | 定义本地base训练、模型、数据、优化器和增强 | 实验配置，不是独立网络模块 |
| `openworld_1/standard_ovd_components.py` 的 `StandardYOLOWorldHead.loss` | 用 `unpack_gt_instances` 接入标准 MMDetection 数据样本，调用原检测损失 | 接口适配，不能单独声称提升多少 AP |
| 同文件 `PositiveDetectionLossHook` | 前25 iter 检查分类/回归/DFL损失有限，且回归与DFL出现正值 | 防止无效训练的检查，不参与推理 |
| `train_with_registered_mmyolo.py` / `test_with_registered_mmyolo.py` | 先注册 MMYOLO 模块，再执行原 `tools/train.py` / `tools/test.py` | 启动兼容性处理 |
| `fixedwing_adapt/baseline.py`、`baseline.sh` | 将测试扩到三类，复测base权重 | 评估配置与Slurm脚本，不训练 |

## 4. 阶段B：用少量新类别标注提升 AP50

### 4.1 方法不是换一个大网络，而是受控微调原 neck/head

方法借鉴 [TFA/FsDet](https://github.com/ucbdrive/few-shot-object-detection) 的“两阶段适配、冻结大部分模型、平衡新旧类样本”思想，但**不是直接复制 TFA，也没有把 Faster R-CNN 模块插进 YOLO-World**。

实际落地在以下几处：

| 操作 | 代码位置 | 原理/目的 |
|---|---|---|
| 冻结图像和文本编码器 | `standard_ovd_components.py::FrozenBackboneYOLOWorldDetector` | 少量样本下保留预训练表征，限制需要拟合的参数 |
| 只允许 neck/head 更新 | `fixedwing_adapt/components.py::FewShotYOLOWorldDetector` + 配置 | 调整文本引导特征融合、类别匹配和边框预测，使模型适应本地飞机外观 |
| 冻结所有 BN running statistics | `FewShotYOLOWorldDetector.train()` 中对 BN 调用 `eval()` | batch=4且样本大量重复，避免均值/方差被少量图像带偏；不等于把neck/head中BN的仿射参数全部禁止学习 |
| 审计可训练参数 | `TrainableParameterAuditHook` | 确保可训练参数仅在 `neck.` / `bbox_head.`，避免误把编码器打开 |
| base replay | `prepare_support.py`、`prepare_5shot.py` | 新类训练同时回放旧类，缓解灾难性遗忘 |
| 轻量增强 | `fixedwing_adapt/finetune.py::train_pipeline` | 小角度旋转、尺度/平移、HSV、翻转，避免极少样本下过强拼接增强 |
| novel AP50选模 | `FixedWingCocoMetric` | 从COCO precision张量提取固定翼AP50；不是更改匹配标准或损失函数 |

10-shot运行的参数审计日志记录：可训练 **16,520,286 / 91,804,238 参数，约18.00%，162个张量**。5-shot继承相同网络与冻结策略。总数包含模型中的文本编码器，不应直接当作TensorRT部署引擎参数量。

**为什么5张图也可能大幅提高精度？** 模型早已学到通用视觉特征和飞机相关语言语义；本地少量标注是在调整已有能力的类别边界、场景响应和定位，而不是从5张图从零学习飞机。重复采样和轻量增强提供更多更新机会，验证选模限制过拟合。不过没有对冻结、replay、BN分别做“去掉该项”的完整精度消融，所以不能把40个百分点的提升定量归因于某一个模块。

### 4.2 10-shot：先验证少样本方案可行

从原train中按seed `20260912`抽取10张、每张恰有1个 fixed-wing 实例的图片；排除近重复候选，并冻结清单，不按测试效果重抽。

- 新类：10张独立图片，重复4次，共40条采样记录。
- 旧类：multirotor 20张 + helicopter 20张，共40张独立replay图片。
- 训练清单合计80条，新旧类曝光比例40:40；独立图像数为50，不是80张独立标注。
- 图像1280、batch=4、AdamW `lr=3e-5`、`weight_decay=0.01`、梯度裁剪10。
- 旋转±5°、平移比例0.05、缩放0.8～1.2、水平翻转概率0.5，以及HSV增强；本少样本配置不使用前述基础训练的Mosaic/MixUp。
- 固定学习率，600 iter，每100 iter在原val验证，按 novel AP50保存最佳权重。
- 4090 Slurm：baseline 85511、train 85512、test 85513。

| 本阶段文件/产物 | 职责 |
|---|---|
| `openworld_1/fixedwing_adapt/audit_dataset.py`、`audit/records.json`、`audit/report.json` | 数据内容、像素哈希与近重复审计 |
| `prepare_support.py`、`data/support_manifest.json` | 固定10-shot抽样和replay协议 |
| `data/train_10shot.json`、`data/val.json` | 训练采样与验证COCO清单，引用原图而非复制图像 |
| `components.py` | 冻结BN的检测器和novel AP50评估器 |
| `finetune.py` | 训练策略、学习率、增强、验证频率 |
| `verify_setup.py` | shot数、数据管线、类别映射、合成框AP检查 |
| `train.sh`、`evaluate.sh`、`evaluate_best.py` | 4090训练及验证选模后的test评估 |
| `runs/neck_10shot_85512/` | 展开配置、训练日志、最佳iter500权重 |
| `test_results/neck_10shot_85512/summary.json` | 最终test AP50=0.8299127475及协议 |

`openworld_1/01_baseline.sh`、`02_train_10shot.sh`、`03_evaluate.sh`是方便寻找的脚本入口链接，不是三种新算法。

### 4.3 5-shot：更少新类监督的独立实验，也是最终加速起点

在 `openworld_2` 中，取先前固定10-shot清单的前5张，仍是5个独立实例。**重新加载原base epoch100权重，不加载10-shot微调权重。**

保持其他设置不变，将5张新类图像各重复8次，仍为40条novel + 40条base记录，80条采样记录；独立图像数为45。不能把8次重复说成40-shot。

| 新增文件/产物 | 相对10-shot改变了什么 |
|---|---|
| [prepare_5shot.py](../openworld_2/prepare_5shot.py) | 固定前5个support，检查实例数和像素去重，生成8次重复清单 |
| `openworld_2/data/support_manifest.json` | 精确记录5张文件名、随机种子和独立初始化说明 |
| `openworld_2/data/train_5shot.json` / `val.json` | 改为5-shot训练清单，val内容沿用原划分 |
| [finetune.py](../openworld_2/finetune.py) | 继承10-shot训练实现，仅覆盖数据路径和输出路径，不新增网络模块 |
| [train_5shot.sh](../openworld_2/train_5shot.sh) | 4090作业85544，训练完成后自动调用test |
| `openworld_2/evaluate_best.py` | 测试由val选出的单个最佳权重并保存摘要 |
| `openworld_2/runs/neck_5shot_85544/` | 展开配置、日志、最佳iter400权重 |
| `openworld_2/test_results/neck_5shot_85544/` | 最终test输出，不是训练过程验证结果 |

### 4.4 每100 iter的验证曲线：为什么不取最后一次

下表都是 **val 的 fixed-wing AP50**，不是test，也不是训练loss。数值取两个运行的 `vis_data/scalars.json` 中按顺序出现的6次验证记录。

| 训练迭代 | 10-shot val AP50 | 5-shot val AP50 |
|---:|---:|---:|
| 100 | 34.7821% | 33.4155% |
| 200 | 46.8465% | 53.0831% |
| 300 | 55.9942% | 52.0605% |
| 400 | 58.7389% | **56.5997%（选中）** |
| 500 | **60.4104%（选中）** | 52.7987% |
| 600 | 58.5597% | 52.7641% |
| 对选中权重做一次原test评估 | **82.9913%** | **80.1431%** |

日志来源：[10-shot scalars](../openworld_1/fixedwing_adapt/runs/neck_10shot_85512/20260912_154745/vis_data/scalars.json)、[5-shot scalars](../openworld_2/runs/neck_5shot_85544/20260912_162312/vis_data/scalars.json)。

5-shot 的val最高56.60%、test80.14%并不意味着测试时又微调了。它们是不同图像集合，novel各只有55/56张，场景、尺度和难度不同，AP可以明显不同。这一差异本身既不能证明泄露，也不能证明完全无泄露；应结合第2节的审计和数据局限解释。

## 5. 阶段C：不再训练，只对5-shot模型逐步提速

### 5.1 固定条件和计时口径

加速源权重始终是：

`openworld_2/runs/neck_5shot_85544/best_coco_fixedwing_ap50_iter_400.pth`

- 本机 `dm-server-7`、RTX3090，PyTorch2.0.1/CUDA11.8；最终TensorRT10.2.0.post1/cu11。
- 输入固定 `[1,3,1280,1280]`，三个文本为 `a multirotor drone`、`a helicopter`、`a fixed-wing aircraft`。
- 选100张固定随机真实test图像（seed20260912），3轮，共300次；预热30次后计时。
- **纯推理：**已预处理的GPU输入 → 原始分类和边框输出；CUDA同步墙钟计时。Graph输入缓冲区复制计入；图像预处理、FP16/layout转换、后处理不计入。
- **端到端：**已解码BGR内存图像 → resize/letterbox、传输和归一化 → 前向 → 原解码/NMS → CPU检测框。不含磁盘读取、JPEG解码、模型加载和引擎构建。
- 55%按 **fixed-wing AP50** 验收，10ms按 **平均纯推理** 验收，不要求每帧/P99绝对小于10ms。

训练日志中的 `time` 含数据读取、训练或整个验证循环，不能直接当作纯推理耗时。先前对话中的约26.54ms历史纯推理值，其旧测速目录当前未保留在本阶段结果中；本文主要采用当前JSON可复核的24.048ms缓存文本FP32作为速度对照，不将二者差值直接认定为严格的缓存消融收益。

### 5.2 逐阶段实测总表

这里的每行都是相同5-shot权重的部署变体；“未单独测”不能用相邻行精度填补。TensorRT是从FP32模型导出的一条部署分支，不是把所有PyTorch优化顺次叠加到最终引擎。

| 步骤/variant | 新加入的处理 | 平均前向ms | FPS | fixed-wing test AP50 |
|---|---|---:|---:|---:|
| S1 `cached_fp32` | 固定文本特征，不每张图运行CLIP | 24.048 | 41.58 | 80.1602% |
| S2 `cached_fp16` | 在S1上转FP16 | 13.231 | 75.58 | 未单独测 |
| S3 `fused_fp16` | 再融合Conv/BN | 12.507 | 79.95 | 未单独测 |
| S4 `fused_fp16_graph` | 再加入CUDA Graph | 12.146 | 82.33 | 未单独测 |
| S5 `fused_fp16_channelslast_graph` | 再使用channels-last内存布局 | 10.752 | 93.01 | 未单独测 |
| S6 `reparam_fp16_channelslast_graph` | 再将固定类别分类计算折叠成3输出卷积 | 10.367 | 96.46 | 80.1057% |
| S7 `reparam_neck_fp16_channelslast_graph` | 再将neck固定文本注意力改成分组卷积 | 10.354 | 96.58 | 未单独测 |
| S8 `tensorrt_fp16` | FP32固定文本+分类头折叠+BN融合 → ONNX → TRT FP16 | 6.305 | 158.60 | 未单独跑全test |
| S9 `tensorrt_fp16_graph` | S8引擎加CUDA Graph | **6.020** | **166.10** | **79.8859%** |

S7相比S6只快约0.013ms，没有做统计显著性验证，不能声称neck卷积化带来稳定大幅加速。S1～S7均未达到10ms目标；S9才完成速度与精度联合验收。

### 5.3 S1：缓存固定词表，移出每帧文本编码

**作用：**类别不变时，CLIP对相同文本的输出无需重复计算。初始化时计算一次，把文本特征作为模型buffer保存，逐帧只执行图像backbone、neck和head。

**新增代码：**[accelerate.py](accelerate.py) 中的 `build()`、`FixedVocabulary`。这不是学习新prompt，没有梯度更新。缓存后的FP32输出与原多模态前向在抽样图像上通过 `atol=rtol=1e-5` 检查；全test另行复测AP50=80.1602%。

**输出：**`results/latency.json` 的 `cached_fp32`、`results/cached_fp32_accuracy.json`、`results/cached_fp32_predictions.json`、`fp32_accuracy.log`。80.16%与原80.14%的细小差异应归入复测数值/执行路径差异，不能解释成新增监督产生的提升。

### 5.4 S2：FP16半精度

**作用：**模型和输入转半精度，降低访存量并利用GPU适合的半精度计算内核。没有剪枝或减少层数。

**代码：**同一 `accelerate.py` 中的 `engine.half()` 和输入dtype转换；本步骤没有新增第二个脚本。平均24.048→13.231ms，约减少45.0%。后处理将输出转回FP32再做解码/NMS，避免不必要的半精度后处理误差。

**输出：**`results/latency.json::cached_fp16`、`experiment.log`。没有单独全test精度，不能说该行AP完全不变。

### 5.5 S3：Conv-BN推理融合

**作用：**推理时BN使用固定均值/方差，可以吸收到卷积权重和偏置，减少独立BN算子与中间读写。

对 `y=W*x+b`，BN缩放系数为 `a=gamma/sqrt(var+eps)`，融合后 `W'=aW`、`b'=a(b-mean)+beta`。这是固定统计量下的代数合并，不是重新学习参数。

**代码：**`accelerate.py` 调用 `mmcv.cnn.fuse_conv_bn`。平均13.231→12.507ms；输出仍在 `latency.json::fused_fp16`，无单独新增脚本。

### 5.6 S4：CUDA Graph

**作用：**固定尺寸的计算序列预先捕获成CUDA Graph，后续重放，减少Python/CPU逐算子启动开销；并不跳过网络层。

**代码：**`accelerate.py::CapturedForward`，维护固定输入缓冲区和输出缓冲区，先预热再捕获。平均12.507→12.146ms。

**输出：**`latency.json::fused_fp16_graph`。输出buffer会被下一次调用覆盖，同一对象不支持无保护并发；更换形状或词表要重建/重捕获。

### 5.7 S5：channels-last内存格式

**作用：**逻辑形状仍是BCHW，只调整内存布局，让FP16卷积更适合某些cuDNN/Tensor Core内核；不是改变RGB/BGR或图像空间分辨率。

**代码：**`accelerate.py` 的 `torch.channels_last` 分支。平均12.146→10.752ms，仍未达标。

**输出：**`results/latency_fused_fp16_channelslast_graph.json`、`channels_last.log`。本步骤仍复用同一脚本，未新增网络层。

### 5.8 S6：分类头固定词表重参数化

**作用：**原检测头先做视觉1×1投影，得到512维视觉嵌入，再经BN与三个固定文本向量做匹配。词表固定且推理BN统计量固定时，这串线性/仿射操作可以合成直接输出3类logit的1×1卷积，减少大维度中间结果。

示意公式：原输出 `s = L × BN(Wx+b) + c`，其中 `L` 为归一化文本乘以learned logit scale；写BN为 `Az+d` 后，可合并为 `s=(LAW)x + L(Ab+d)+c`。代码没有改变前面的分类分支卷积、边框回归分支或类别语义。

**代码：**`accelerate.py::fold_classifiers`、`AlreadyClassified`；逐一替换三尺度 `head.head_module.cls_preds[i][-1]` 和 `cls_contrasts[i]`。BNContrastiveHead的结构使这里可以做这种折叠；不能不检查模型类型就推广到任意带非线性归一化的头。

**结果：**FP32折叠前后抽样输出通过 `1e-4` 误差容限检查；转FP16、融合、channels-last及Graph后平均10.367ms，全test AP50=80.1057%。

**输出：**`results/latency_reparam_fp16_channelslast_graph.json`、`results/reparam_fp16_channelslast_graph_accuracy.json`、相应 `_predictions.json`、`reparameterize.log`、`native_accuracy.log`。

### 5.9 S7：neck固定文本注意力卷积化（探索分支）

**作用：**原 `MaxSigmoidAttnBlock` 每次将固定文本经 `guide_fc` 映射，再与图像嵌入做 `einsum` 匹配。将固定guide预计算并写入分组1×1卷积，仍保留类别max、sigmoid和原特征门控语义。

**代码：**`accelerate.py::FixedAttention`、`fold_attention`。它不是增加新注意力机制，而是对原注意力的固定词表实现方式做改写。

**结果/输出：**平均10.354ms，仍未达标；`results/latency_reparam_neck_fp16_channelslast_graph.json`、`neck_reparam.log`。没有为此分支独立跑全test。

**重要：最终TensorRT脚本没有调用 `fold_attention`。** 不能把S7说成最终6.02ms必不可少的模块，也不能把channels-last看作最终TensorRT引擎的人工输入格式；TensorRT分支使用连续FP32输入，自行优化内部布局。

### 5.10 S8：ONNX导出与TensorRT FP16引擎

**作用：**PyTorch常规执行未达到10ms后，采用TensorRT离线优化计算图、融合算子、选择适合本机GPU的FP16内核。使用FP32模型导出，构建时允许FP16策略，输入/输出保留FP32；不是INT8量化，也没有INT8校准数据。

**新增脚本：**[tensorrt_experiment.py](tensorrt_experiment.py)。

- `ExportRaw`：导出三尺度类别和边框原始输出，共6个输出；原解码/NMS留在引擎外。
- `build_engine`：`fold_classifiers` + `fuse_conv_bn`，静态ONNX opset13导出并检查；TensorRT10.2构建，FP16启用、TF32关闭、workspace2GiB、optimization level3。
- `TensorRTEngine`：加载序列化引擎、预分配输出GPU缓冲区、设置地址并调用 `execute_async_v3`。
- 引擎manifest：记录源权重SHA256、文本、输入形状、GPU和TensorRT版本，避免误用不同模型/设备的引擎。

| 新增产物 | 内容 |
|---|---|
| `artifacts/fixed_vocab_1280.onnx` | 固定词表和形状的导出图 |
| `artifacts/fixed_vocab_1280_fp16.engine` | 实际部署的TensorRT引擎 |
| `artifacts/engine_manifest.json` | 源模型与设备的身份检查信息 |
| `artifacts/timing.cache` | 引擎构建的算子计时缓存，不是模型权重 |
| `requirements-deploy.txt`、`vendor/` | ONNX1.14.1、protobuf3.20.3、TRT10.2/cu11的本地依赖，不覆盖训练环境 |
| `run_tensorrt.sh` | 设置本地依赖与离线环境，统一测速/测试/图片推理入口 |
| `tensorrt.log` | ONNX检查、引擎构建和首次测速日志 |

普通TRT前向平均6.305ms。首轮普通TRT使用默认stream有额外同步提示；Graph版本单独实测，报告没有将该警告隐藏成理论最优性能。引擎构建约142.5秒，是一次性离线开销，不计入6ms推理值。

### 5.11 S9：TensorRT + CUDA Graph，最终联合验收

**处理：**复用 `CapturedForward` 捕获TensorRT执行。没有再改权重或新增模型层。对同一个引擎进行完整test，并独立启动第二次速度复测。

| 指标 | 首次测量 | 独立复测 |
|---|---:|---:|
| 平均纯推理 | **6.020ms** | **5.936ms** |
| 纯推理FPS | 166.10 | 168.47 |
| 纯推理P99 | 6.340ms | 6.186ms |
| 纯推理最大实测值 | 6.419ms | 6.277ms |
| 平均端到端 | 9.715ms | 9.470ms |
| 端到端FPS | 102.93 | 105.60 |

这里保留首次测量作为主结果，不挑最快一次替换。相对当前缓存FP32对照24.048ms，主结果约 **3.99倍速度、减少75.0%平均耗时**。这些是给定硬件和测试流程下的实测，不是所有部署环境的硬实时保证；端到端仍有超过10ms的帧。

最终fixed-wing AP50=**0.7988587917**，相比原5-shot test 0.8014306996下降约 **0.257个百分点**，远高于0.55；相比基线复测0.3947218898提高约 **40.414个百分点**。这是少样本学习带来的主要精度提升经部署转换后得到保留，不是TensorRT使AP从39%升到80%。

## 6. 最终完整类别指标与AP的含义

以下来自 [tensorrt_fp16_graph_accuracy.json](results/tensorrt_fp16_graph_accuracy.json)，数值保留三位小数：

| category | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|---|---:|---:|---:|---:|---:|---:|
| multirotor drone | 0.512 | 0.953 | 0.521 | 0.351 | 0.518 | 0.626 |
| helicopter | 0.566 | 0.958 | 0.538 | nan | 0.033 | 0.568 |
| fixed-wing aircraft | 0.311 | 0.799 | 0.166 | nan | 0.514 | 0.292 |

- AP50只要求IoU≥0.50；AP75要求更精确定位。飞机细长、框边缘误差或角度差异可让IoU明显变化，因此高AP50不意味着高AP75；具体误差来源仍需逐框分析，不能仅从表格确定。
- `mAP`是IoU=0.50、0.55、…、0.95的平均。最终fixed-wing是31.13%，**并没有达到55%的COCO mAP**，达标的是前文约定的AP50。
- `mAP_s/m/l`按COCO小/中/大目标面积分组，仍对多个IoU阈值取平均，不是AP50的大小分组，也不是应该加起来等于AP50的组成部分。COCO常用面积界限为32²和96²像素，依据评估标注面积，不应按输入缩放后的框重新解释。
- `nan`/JSON `null`表示该面积组没有可评估的对应真值，不表示AP为0。
- FP16改变舍入、分数和NMS排序，某些类别AP可能略升、略降；没有逐框完全相等的保证。精度验收依靠完整938张测试，而不是只看导出成功。

## 7. openworld_3 文件导航：找结果、找模块、找可执行入口

| 路径 | 该看什么 |
|---|---|
| [summary.md](summary.md) | 本文，完整历史路线、解释和证据 |
| [RESULTS.md](RESULTS.md) | 简短速度消融及最终类别表 |
| [README.md](README.md) | 环境、复现命令、使用限制 |
| [accelerate.py](accelerate.py) | 固定文本、分类/注意力折叠、PyTorch测速和COCO评估 |
| [tensorrt_experiment.py](tensorrt_experiment.py) | ONNX/TRT构建、TRT测速、全test、单图推理 |
| [run_tensorrt.sh](run_tensorrt.sh) | 实际执行入口 |
| [summarize.py](summarize.py) | 从已产生JSON汇总 `RESULTS.md` 和 `results/acceptance.json`；不训练、不测速，不自动重写本文 |
| [results/tensorrt_latency.json](results/tensorrt_latency.json) | 首轮300次原始耗时、统计、环境和FP16输出误差 |
| [results/tensorrt_latency_repeat.json](results/tensorrt_latency_repeat.json) | 独立复测耗时 |
| [results/tensorrt_fp16_graph_accuracy.json](results/tensorrt_fp16_graph_accuracy.json) | 最终精度验收依据 |
| `results/tensorrt_fp16_graph_predictions.json` | 完整测试检测框，可进一步检查误检漏检/定位 |
| `results/acceptance.json` | 各变体数据与双指标通过状态 |
| `results/inference.json`、`inference_smoke.log` | 保存引擎的单图片推理检查输出，不是另一个训练实验 |
| `artifacts/` | 当前保留ONNX、engine、manifest和timing cache |
| `native_accuracy.log`、`fp32_accuracy.log`、`tensorrt_accuracy.log` | 各完整精度复测日志 |
| `experiment.log`、`channels_last.log`、`reparameterize.log`、`neck_reparam.log` | PyTorch各步执行记录 |
| `tensorrt.log`、`tensorrt_repeat.log` | 最终引擎与两轮速度日志 |
| `rescore.log` | 按原配置评估参数对已保存预测的重新评分记录，不更新权重 |
| `aaa_speed/` | 之前已有目录，不是这条加速管线必需模块 |

代码有保存中间PyTorch `.pth` 的逻辑，但本文整理时 `artifacts/` 当前只保留上述四类TRT相关文件；不要把中间权重当作当前必定存在的交付物。消融结果JSON仍保留，可以按脚本重跑。

原PyTorch消融的端到端测量只预热前向，含首次后处理初始化离群值；不拿它们与最终TRT端到端直接宣称精确加速倍数。最终TRT对前向和后处理都预热30次。

## 8. 复现与依赖边界

从项目根目录执行已有部署引擎的操作：

```bash
cd /home/songziqi/project/YOLO-World

# 构建/复用引擎并测速；默认会更新首次测速JSON
bash openworld_3/run_tensorrt.sh

# 若希望保留现有主测速结果，指定一个新的输出文件
bash openworld_3/run_tensorrt.sh --latency-output openworld_3/results/tensorrt_latency_new.json

# 对验证集已经选好的同一权重/引擎跑完整原test
bash openworld_3/run_tensorrt.sh --evaluate

# 自己的单张图片，输出框为xyxy，label顺序对应三个固定文本
bash openworld_3/run_tensorrt.sh --image /absolute/path/image.jpg --output openworld_3/results/my_image.json

# 根据现有JSON刷新简短报告，不重新做实验
/home/songziqi/anaconda3/envs/YOLO-World/bin/python openworld_3/summarize.py
```

需要重新训练5-shot时才提交 `sbatch openworld_2/train_5shot.sh`，它会生成新job目录。当前 `accelerate.py::CONFIG/CHECKPOINT` 固定指向85544的已选模型；新训练不会自动替换加速源模型。变更源权重、类别、输入形状或GPU/TensorRT环境后，应保留旧产物、另行导出ONNX和构建engine，不能直接复用当前engine。

当前推理wrapper仍初始化原PyTorch模型以复用预处理/后处理，并依赖 `openworld_2` 的展开配置、权重及 `openworld_1` 的配置/自定义类。这是可运行的本项目部署入口，**不是完全脱离原仓库的单文件独立程序**。因此不要仅保留engine后删除这些依赖；模型初始化不包含在稳态6.02ms中。

## 9. GitHub方法对应关系：采用了什么，没有采用什么

| 项目/资料 | 本实验采用程度 |
|---|---|
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) | 原模型、预训练体系和代码基础 |
| [YOLO-World reparameterize说明](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/reparameterize.md) | 借鉴固定文本及卷积重参数化；本地实现按当前BNContrastiveHead结构扩展了分类投影融合 |
| [YOLO-World deploy说明](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/deploy.md) | 参考导出路线；不能声称官方文档已提供本项目可直接运行的全部TRT代码 |
| [TFA / FsDet](https://github.com/ucbdrive/few-shot-object-detection) | 借鉴冻结与平衡少样本适配思想，不是原架构逐字复现 |
| [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT) | 实际安装并构建10.2/cu11引擎，最终速度达标的关键执行后端 |
| [PyTorch CUDA Graph](https://github.com/pytorch/pytorch/blob/main/torch/cuda/graphs.py) | 使用本机torch2.0.1接口捕获/重放，不是复制最新main分支 |
| [channels-last教程](https://github.com/pytorch/tutorials/blob/main/intermediate_source/memory_format_tutorial.py) | 实际PyTorch内存布局消融；最终TRT自行优化内部布局 |
| [YOLO-World prompt tuning](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/prompt_yolo_world.md) | 研究备选，未训练learned prompt/adapter |
| [DeFRCN](https://github.com/er-muyue/DeFRCN)、[DCFS](https://github.com/gaobb/DCFS) | 研究备选，未移植、未安装相关检测模块，也没有本地提点数据 |

本实验没有采用知识蒸馏、剪枝、INT8量化、换小模型或降低输入分辨率；不能在报告里把这些常见加速方法列为已经实现。

## 10. 可直接用于实验描述的一段话

本实验以公开YOLO-World V2-M 1280预训练权重为起点，在两个基础类别上进行本地训练，通过基础验证集选出epoch100模型，其在固定三类测试集上的fixed-wing AP50为39.47%。随后独立进行5-shot和10-shot新类别适配，冻结图像/文本编码器及BN运行统计，仅优化原neck和head，并使用40张基础类别样本回放；利用额外55张novel验证标注选模后，5-shot和10-shot的测试AP50分别达到80.14%和82.99%。部署阶段选择5-shot模型，采用固定文本嵌入、分类头重参数化、Conv-BN融合及TensorRT FP16/CUDA Graph，在RTX3090、batch1、1280输入下平均纯推理6.02ms、166.10FPS，完整938图测试fixed-wing AP50为79.89%，满足平均纯推理低于10ms且AP50不低于55%的目标。该结果沿用原始数据划分，其base类跨划分重叠及可能的同源相关性仍是实验限制。
