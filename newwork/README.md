# Sparse-Zoom YOLO-World：文本引导的稀疏高分辨率无人机检测

本目录是独立插件式实现，不修改 `yolo_world/` 上游源码。方法在原
YOLO-World P3–P5 PAN 前保持全部预训练参数名不变，新增 stride-4 P2
检测层，并只对文本相关的 Top-K 高分辨率窗口执行局部精炼。

## 目录

```text
newwork/
├── models/       # P2 稀疏 neck、动态/rep router、router 辅助损失
├── configs/      # 动态训练配置与重参数化测试配置
├── tools/        # 权重迁移、重参数化、MMYOLO 注册入口
├── slurm/        # 训练与 rep+测试作业
├── tests/        # router 和四尺度构建测试
├── docs/         # 方法、实验计划与消融计划
└── WORKLOG.md    # 本次实现记录与验证结果
```

## 核心设计

1. Backbone 输出 P2/P3/P4/P5；原 `YOLOWorldPAFPN` 原样处理 P3–P5。
2. 原始 P2 与上采样 P3 融合，产生 1/4 分辨率检测特征。
3. 图像区域向量和文本向量投影并归一化，点积产生文本显著图。
4. 显著图按窗口池化，每幅图选 Top-K；共享的 depthwise refinement
   只运行在被选窗口，再散射回 P2。
5. 新 head 使用 `[4, 8, 16, 32]` 四个 stride。router 通过目标区域
   focal loss、GT coverage loss 和软稀疏预算 loss 联合训练。

“稀疏”特指新增的 P2 局部精炼分支；当前实现没有把整个 backbone
改成稀疏卷积。论文中应分别报告完整模型 latency/FLOPs，以及 P2
refinement 的理论激活比例，不能把 25% 窗口误写成全模型节省 75%。

## 训练

默认数据使用已有的联合 COCO 标注：

- `drone-yolo-rgb+UAV1/annotations/instances_{train,val,test}.json`
- 单类别和文本提示均为 `drone`
- 输入 1280，YOLO-World V2-M，默认 Top-25% 的 8×8 P2 窗口

提交作业：

```bash
cd /home/songziqi/project/YOLO-World
sbatch newwork/slurm/train_sparse_zoom.sh
```

可覆盖超参数：

```bash
RUN_TAG=ablation_r0125 BATCH=2 SPARSE_RATIO=0.125 WINDOW_SIZE=8 \
  sbatch newwork/slurm/train_sparse_zoom.sh
```

脚本首次运行会将三尺度预训练 head 的 P3/P4/P5 参数从索引 0/1/2
迁移到四尺度 head 的 1/2/3；新 P2 head、P2 fusion 和 router 随机初始化。
原 neck 权重不重排。

## 重参数化与测试

动态文本模型训练完成后再转换。假设最佳权重为：

```bash
CHECKPOINT=/home/songziqi/project/YOLO-World/newwork/runs/\
sparse_zoom_v2_m_drone_1280_e80/best_coco_bbox_mAP_epoch_80.pth \
  sbatch newwork/slurm/reparameterize_and_test.sh
```

转换会自动发现四个 contrastive head、四个原 PAN 文本注意力块，并将
新 P2 router 的 `text_proj` 写入固定 `guide_conv`。转换后的模型无需在
forward 时输入文本 embedding。

## 快速验证

```bash
export PYTHONPATH=/home/songziqi/project/YOLO-World:$PYTHONPATH
/home/songziqi/anaconda3/envs/YOLO-World/bin/python \
  newwork/tests/test_model_build.py
```

完整方法与论文实验设计见 [METHOD.md](docs/METHOD.md)、
[EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) 和
[ABLATION_PLAN.md](docs/ABLATION_PLAN.md)。

