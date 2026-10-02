"""Create a compact report from measured artifacts, never estimated timings."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
results=HERE/'results'
rows={}
for path in sorted(results.glob('latency*.json')):
    for name,data in json.loads(path.read_text())['results'].items():
        rows[name]=dict(data)
trt=results/'tensorrt_latency.json'
if trt.exists():
    rows.update(json.loads(trt.read_text())['results'])
for name,row in rows.items():
    accuracy=results/f'{name}_accuracy.json'
    if accuracy.exists():
        row['accuracy']=json.loads(accuracy.read_text())['metrics']
    ap=row.get('accuracy',{}).get('fixed-wing aircraft',{}).get('AP50')
    row['joint_target_met']=row['forward']['mean_ms']<10 and ap is not None and ap>=.55

lines=['# 推理加速实验实测结果','',
       '验收：RTX 3090，batch=1，1280×1280，平均纯推理 <10 ms，fixed-wing aircraft 测试 AP50 ≥55%。',
       '同一 5-shot iter400 权重、三个文本类别与原始 938 张测试图像；没有重新训练或降低分辨率。','',
       '| 方法 | 平均前向 ms | FPS | P99 ms | 最大 ms | fixed-wing AP50 | 双指标通过 |',
       '|---|---:|---:|---:|---:|---:|---|']
for name,row in rows.items():
    f=row['forward']
    ap=row.get('accuracy',{}).get('fixed-wing aircraft',{}).get('AP50')
    ap_text=f'{ap*100:.2f}%' if ap is not None else '未单独验证'
    lines.append(f"| {name} | {f['mean_ms']:.3f} | {f['fps']:.2f} | {f['p99_ms']:.3f} | {f['max_ms']:.3f} | {ap_text} | {'是' if row['joint_target_met'] else '否/待核验'} |")
repeat=results/'tensorrt_latency_repeat.json'
if repeat.exists():
    r=json.loads(repeat.read_text())['results']['tensorrt_fp16_graph']
    lines+=['',f"独立速度复测：平均纯推理 {r['forward']['mean_ms']:.3f} ms / {r['forward']['fps']:.2f} FPS；端到端 {r['end_to_end']['mean_ms']:.3f} ms。主表保留首次完整测量，没有挑选最快一次替换。"]
if 'tensorrt_fp16_graph' in rows:
    selected=rows['tensorrt_fp16_graph']
    e=selected['end_to_end']
    lines+=['','## TensorRT 端到端与精度','',
        f"预处理→前向→原始解码/NMS→CPU 检测框：平均 {e['mean_ms']:.3f} ms，{e['fps']:.2f} FPS。",
        '不包括磁盘读取、JPEG 解码和模型初始化；已预热前向与后处理。','',
        '| 类别 | mAP | mAP50 | mAP75 | mAP_s | mAP_m | mAP_l |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in selected.get('accuracy',{}).items():
        values=[m[k] for k in ['mAP','AP50','AP75','mAP_s','mAP_m','mAP_l']]
        lines.append('| '+name+' | '+' | '.join('nan' if v is None else f'{v:.3f}' for v in values)+' |')
lines+=['','## 口径与限制','',
    '- 纯推理从已预处理 GPU tensor 到原始分类/回归输出；含同步等待，CUDA Graph 还包含输入缓冲区复制。',
    '- 100 张固定随机真实测试图像 ×3 轮，30 次预热；平均达标不代表每一帧都低于 10 ms。',
    '- 原 PyTorch 消融的端到端记录含首次后处理初始化离群值，不用于最终端到端结论。',
    '- mAP 是 IoU 0.50～0.95 的平均，55% 指标按前文的 fixed-wing mAP50 验收。',
    '- FP16 会带来数值与 NMS 排序差异，报告完整测试集的实际结果，不声称逐框等同。',
    '- 原始数据集的重叠问题仍存在；本实验是同一测试集上的部署前后比较，不代表全新无泄露实验。',
    '- TensorRT engine 与 GPU/运行库绑定；固定类别、固定尺寸，换类别或设备需要重新构建。','',
    '复现命令、原项目 GitHub 链接见 [README.md](README.md)。原始逐次耗时与精度见 `results/`。']
(HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(results/'acceptance.json').write_text(json.dumps(rows,indent=2))
print(HERE/'RESULTS.md')
