# Sparse-Zoom openworld_base 训练

本目录按照 `HANDOFF.md` 的 base-to-novel 训练方式，使用 Obj365/GoldG
YOLO-World V2-M 权重初始化四尺度 Sparse-Zoom 模型，并只在
`multirotor drone`、`helicopter` 两个 base 类上微调 80 epoch。

- 输入：1280
- 默认 batch：2（P2 高分辨率分支比原模型更占显存）
- AdamW：沿用原配置的 `2e-4`
- 每 5 epoch 验证；最后 10 epoch 关闭 Mosaic
- `fixed-wing aircraft` 不参与训练
- 输出保持动态文本形式，不自动重参数化，以便后续 zero-shot/5-shot

提交：

```bash
cd /home/songziqi/project/YOLO-World
sbatch newwork/openworld_1/train_sparse_zoom_ovd_base2_e80.sh
```

输出目录：

```text
newwork/openworld_1/
├── weights/   # 三尺度到 P2--P5 的初始化 checkpoint
├── runs/      # 80 epoch 训练结果
└── slurm_logs/
```

## 当前运行

- Slurm job：`79997`
- run tag：`sparse_zoom_v2_m_ovd_base2_1280_e80`
- 提交时间：2026-08-24
- 初始化审计：728 个原权重张量全部同形状加载；新 P2/router 按设计初始化
- 首个日志点：epoch 1 iter 50，bbox/DFL/router loss 均为有限正值
- 4090 显存：约 18.9 GB（batch 2）

日志：

```text
newwork/openworld_1/slurm_logs/yw_sp_ovd80-79997.out
newwork/openworld_1/slurm_logs/yw_sp_ovd80-79997.err
```

首次提交 `79996` 因 Slurm 日志目录尚不存在而在脚本执行前退出，没有
启动训练、没有写入 work_dir。目录创建后重新提交为 `79997`。

## 后续评测

- 三类开放词表测试与原模型对比：`OPENWORLD3_COMPARISON.md`
- UAV/RGB 5-shot + base replay 对比：`UAV_5SHOT_COMPARISON.md`

UAV 5-shot 使用作业 `80102`，val 最佳点为 iter 200，最终 test mAP50 为
`0.463`。首次提交 `80101` 同样因为新建的 Slurm 日志目录尚不存在而在脚本
执行前退出，没有启动训练或污染 run 目录。
