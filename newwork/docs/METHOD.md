# 方法说明

## 1. 问题与动机

无人机目标在 1280 图像中常只有数个到数十个像素。原 YOLO-World
最细检测层为 stride 8，微小目标的纹理在进入 head 前容易丢失；直接
给整幅图加入 stride-4 重型分支又会产生较高显存与延迟。本文方法用
文本提示定位潜在目标区域，把额外高分辨率计算集中在少量窗口。

## 2. 保留预训练语义的四尺度 PAN

Backbone 输出 `{P2, P3, P4, P5}`。`P3–P5` 仍进入原 YOLO-World
PAFPN，参数名和计算顺序均不变。设原 neck 的 P3 输出为 `N3`，新增
P2 特征为：

```text
F2 = CSP([Conv1x1(P2), Up(Conv1x1(N3))]).
```

最终 head 输入是 `{F2_refined, N3, N4, N5}`，strides 为
`{4, 8, 16, 32}`。这种外围扩展避免了四尺度重建 PAN 导致的预训练
neck 层索引错配。

## 3. 文本引导路由

对 P2 特征 `F2` 和 N 个文本向量 `T` 分别投影、L2 归一化：

```text
E(x,y) = normalize(W_i F2(x,y))
G_n    = normalize(W_t T_n)
S_n(x,y) = exp(tau) * <E(x,y), G_n> + b
S(x,y)   = max_n S_n(x,y).
```

`S` 是 prompt-agnostic 的区域显著图；多提示时 max 聚合保留最相关的
提示。单类实验中 N=1，提示为 `drone`。

## 4. 稀疏窗口精炼

将 `S` 划分为非重叠的 `w×w` P2 窗口，以窗口平均分数排序，选取
`K = ceil(r * L)` 个窗口。只将这 K 个窗口打包为小 batch，运行共享
depthwise-separable residual block，随后 scatter/fold 回原 P2 位置。

Top-K 是硬计算路由；被选窗口的 refinement delta 还乘以其 sigmoid
软门控，使检测损失能向 router 的选中区域回传。未选区域保持融合 P2
原值。delta 最后一层采用零初始化，使新模型初始行为接近普通四尺度
模型，降低训练初期扰动。

## 5. Router 监督

训练总损失为：

```text
L = L_yolo + lambda_r L_focal + lambda_c L_coverage
             + lambda_b L_budget.
```

- `L_focal`：以小目标扩张框区域作为 P2 文本显著图的正样本。
- `L_coverage`：同时惩罚小目标区域内最大路由概率过低，以及 GT 中心
  窗口分数低于当前 Top-K 阈值，直接减少硬路由漏选。
- `L_budget`：约束全图平均软激活概率接近目标稀疏率 r。

默认只监督等效边长不超过 96 像素的目标，和本工作的微小无人机目标
定位一致；检测 head 仍学习所有合法 GT。

## 6. 动态与重参数化

训练时保留 `text_proj: Linear(512, d)`，从 CLIP `drone` embedding
继承开放词汇语义。训练后计算：

```text
G = normalize(T W_t^T + b_t)
guide_conv.weight = G[:, :, None, None].
```

同时转换四层 contrastive head 和原 PAN 的 `guide_fc`。部署模型只需
图像输入。若未来研究多类/开放词汇动态提示，应保留动态版本，不能用
固定单 prompt 的 rep 模型做开放词汇结论。

## 7. 论文可陈述的贡献边界

可主张的是：文本条件区域路由、预训练兼容的 P2 扩展、router 的小目标
覆盖监督，以及动态到固定提示部署转换。是否优于 dense P2、普通
saliency router 或 tile/crop 方法，必须由消融和真实延迟实验支持。
