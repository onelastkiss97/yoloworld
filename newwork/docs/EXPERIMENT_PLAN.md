# 实验计划

## 主实验

统一训练 80 epoch、输入 1280，并固定数据划分与随机种子。至少报告：

| 维度 | 指标 |
|---|---|
| 检测精度 | COCO mAP、AP50、AP75、APs；drone recall |
| 微小目标 | 按目标面积/等效边长分桶的 AP 与 recall |
| 效率 | 参数量、完整 FLOPs、峰值显存、batch=1 latency、FPS |
| 路由质量 | GT window recall、无 GT 激活率、实际窗口比例 |

主要对比：

1. 原三尺度 YOLO-World V2-M。
2. Dense P2 四尺度模型（所有 P2 窗口精炼）。
3. 随机 Top-K P2 路由。
4. 图像-only saliency Top-K。
5. 本文文本引导 Top-K。

为避免只对一个训练域过拟合，建议在联合训练集报告 in-domain test，
并在 openworld 全映射为 `drone` 的数据上报告跨域 zero-shot/transfer
结果。动态开放词汇能力与单类别 rep 部署结果应分表呈现。

## 公平性设置

- 所有对比使用相同预训练 checkpoint、增强、epoch 和输入尺寸。
- Dense P2 与 sparse P2 使用同一 P2 fusion/refinement 参数。
- latency 需预热后测量，分别报告 PyTorch FP32、FP16（若使用）和硬件。
- 同时给出平均值和至少 3 个随机种子的标准差。
- 不能只报告理论 Top-K 比例；需要测完整端到端 latency。

## 建议执行顺序

1. 先跑原模型、普通四尺度和本方法 25% 三组，确认增益来源。
2. 再扫稀疏率与窗口尺寸，确定 Pareto 点。
3. 做 router loss 和文本/图像路由消融。
4. 最后做跨域、提示鲁棒性和真实部署测速。

## 失败诊断

- GT window recall 低：增大 coverage 权重或稀疏率，检查坐标缩放。
- soft 激活全高但硬 Top-K 固定：提高 budget 权重，检查 router logits。
- P2 AP 无提升：比较 dense P2，判断瓶颈在 P2 表征还是路由。
- 延迟无下降：窗口 gather/scatter 开销可能抵消小卷积节省，应增大
  refinement 深度或实现 CUDA/连续区域合并后再讨论部署优势。

