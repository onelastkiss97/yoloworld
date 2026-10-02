# 消融实验清单

## A. 高分辨率分支

| 编号 | P2 | refinement | 目的 |
|---|---:|---|---|
| A0 | 否 | 无 | 原三尺度基线 |
| A1 | 是 | 无 | 仅四尺度 head |
| A2 | 是 | Dense | 精度上界与计算上界 |
| A3 | 是 | Sparse | 完整方法 |

## B. 路由信号

- 随机 Top-K。
- image-only 1×1 saliency。
- 固定 `drone` 文本相似度。
- 多提示：`drone / UAV / small flying object`，max 或 mean 聚合。
- 错误文本负对照，验证模型是否真正使用文本语义。

## C. 稀疏超参数

- 稀疏率：`0.0625, 0.125, 0.25, 0.50, 1.0`。
- P2 窗口：`4, 8, 16`，对应输入空间约 `16, 32, 64` 像素。
- refinement block：`1, 2, 3`。
- router channels：`32, 64, 128`。

主图应画 mAP/GT-window-recall 与 latency/FLOPs 的 Pareto 曲线，而不是
只选择单个最好点。

## D. 损失项

从完整模型依次移除：`L_focal`、`L_coverage`、`L_budget`、检测损失到
router 的软门控梯度。报告每组路由召回、soft/hard 激活率和最终 APs。

## E. 预训练与重参数化

- 原三尺度权重正确迁移 vs 全部随机初始化。
- 动态文本推理 vs 转换后的 rep 推理，检查数值差异和 mAP 差异。
- P3–P5 head 错位加载作为实现 sanity check，不作为正式方法对比。

## F. 建议论文最小结果集

若资源有限，至少完成 A0/A1/A2/A3、三档稀疏率、三种路由信号和损失
移除实验，并补充真实 4090 batch=1 latency。没有这些证据时，“文本
引导”和“稀疏高效”两项核心结论都不够闭环。

