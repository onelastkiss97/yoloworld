# 实现工作记录

## 已完成

- 新建独立 `newwork` 包并注册到 MMYOLO registry。
- 实现动态文本 router 与固定提示 rep router。
- 在不改动原 P3–P5 PAN 参数名的前提下加入 P2 fusion。
- 实现 Top-K 非重叠窗口的真正局部计算、scatter/fold 复原及软门控。
- 新增小目标 focal、coverage、sparse-budget 三项 router loss。
- 将 backbone/head/prior/stride 全部适配到 P2–P5 四尺度。
- 实现原三尺度 checkpoint 的 head 索引迁移。
- 实现四尺度 head、PAN attention 和 P2 router 的自动重参数化。
- 编写动态训练、rep 测试配置及两份 Slurm 脚本。

## 本地验证（2026-08-21）

- Python 静态编译：通过。
- 两份 Slurm shell `bash -n`：通过。
- 动态模型 config build：通过。
- 256×256 smoke forward：输出四尺度空间大小
  `64×64 / 32×32 / 16×16 / 8×8`。
- 动态 router 与 rep router shape 测试：通过。
- router backward：`text_proj` 能收到非零梯度。
- 128×128 完整 detection + router loss + backward：通过。
- Runner、AdamW 配置、联合训练集与 `yolow_collate` 构建：通过；抽取的
  实际 batch 为 `2×3×1280×1280`，标注张量格式为 `N×6`。
- 动态 checkpoint 内存转换到 rep：4 个 head、4 个 PAN attention 和
  1 个 P2 router 全部匹配；rep 模型 strict load 无 missing/unexpected。
- 动态/rep 数值等价：四层 neck 最大绝对误差均为 0，四层分类 logits
  最大绝对误差小于 `5e-8`。
- 原始 checkpoint 迁移规则验证：105 个 head tensor 被移动，neck 未动。

环境未安装 `pytest`，所以测试函数通过 Python 直接调用；没有为此改动
现有环境。尚未提交长时 GPU 训练，训练产物将由 Slurm 写入
`newwork/runs/<RUN_TAG>`。

## 后续运行产物

```text
newwork/
├── weights/       # P2 初始化权重和 rep 权重
├── runs/          # 训练 checkpoint、日志和配置快照
├── test_results/  # test.py 指标输出
└── slurm_logs/    # Slurm stdout/stderr
```
