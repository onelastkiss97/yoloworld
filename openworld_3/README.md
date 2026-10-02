# 5-shot YOLO-World latency optimization

**Measured result:** TensorRT FP16 + CUDA Graph: mean forward **6.020 ms / 166.10 FPS**, fixed-wing test **AP50 79.89%** on 938 images. Both requested targets passed. Warmed end-to-end mean **9.715 ms / 102.93 FPS** (decoded RAM input; not disk/JPEG). See [RESULTS.md](RESULTS.md) for the full table and exact measurement scope.

Goal: **mean** pure forward below 10 ms on the local RTX 3090, batch=1, 1280x1280, and fixed-wing **AP50 >= 0.55** on the unchanged test set. AP50 is not COCO AP averaged over IoU thresholds. Report mean, P95, P99 and maximum observed latency separately; a mean below 10 ms does not imply a per-frame guarantee. The original checkpoint is the validation-selected 5-shot iter400 model from openworld_2. No new training samples, vocabulary, resolution or score/NMS thresholds are changed.

## GitHub research and chosen implementation

- [YOLO-World official reparameterization](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/reparameterize.md): precompute fixed vocabulary embeddings; absorb them into convolution weights. `fold_classifiers` composes the final visual projection, frozen contrastive BN and text classifier into a three-output convolution. `fold_attention` additionally converts fixed-text neck attention into grouped convolutions. FP32 equivalence is checked before FP16 conversion; full test AP is checked separately.
- [YOLO-World ONNX deployment guide](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/deploy.md): exports fixed-vocabulary models; TensorRT instructions in this guide are incomplete. Directly using its legacy exporter also requires attention to current detector interfaces.
- [PyTorch CUDA Graph implementation](https://github.com/pytorch/pytorch/blob/main/torch/cuda/graphs.py): capture and replay fixed-shape CUDA operations, reducing per-operator CPU launch overhead. The experiment uses the installed PyTorch 2.0.1 API, not a code copy from the latest branch.
- [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT): ONNX parsing, FP16 tactic selection and optimized inference engines. TensorRT 10.2/cu11 was built and measured locally; this is the final selected backend. TF32 is disabled.
- [PyTorch channels-last](https://github.com/pytorch/tutorials/blob/main/intermediate_source/memory_format_tutorial.py): memory layout optimization for FP16 convolutions; logical input shape remains BCHW.

## Initial ablation

1. Fixed CLIP text features, FP32.
2. Same fixed features, FP16.
3. Fold inference-time convolution/BatchNorm, FP16.
4. Fused FP16 plus CUDA Graph replay (input buffer copy included in forward timing).

The text prompts stay `a multirotor drone`, `a helicopter`, `a fixed-wing aircraft`. Changing vocabulary requires regenerating the text buffer and recapturing the graph. The fixed graph is batch-one/1280 and not thread-safe for concurrent calls; graph output buffers are reused on subsequent calls.

Measurements use 100 fixed random real test images, three rounds, 30 warmup forwards per variant, CUDA synchronization and wall-clock timing. End-to-end starts from decoded BGR images in RAM and includes resize/letterbox/normalize/H2D, forward, original decode/NMS and CPU output; it excludes disk/JPEG and initialization. Forward excludes preprocessing, dtype conversion and postprocessing. Frozen-text FP32 output is numerically checked against the original multimodal forward.

`accelerate.py` runs timing ablations. `accelerate.py --evaluate --variant fused_fp16_graph` evaluates all 938 original test images with the selected accelerated backend. `results/` contains measured latency and accuracy; `artifacts/` contains deployment weights; `experiment.log` contains runtime output. The pre-existing `aaa_speed/` directory is left untouched.

The original dataset-overlap caveats remain: this is a deployment comparison on the same fixed test set, not a new clean-data generalization experiment.

## TensorRT commands

Dependencies are local to `vendor/`, not installed over the training environment. Run from the repository root:

```bash
bash openworld_3/run_tensorrt.sh                         # build/reuse engine and benchmark
bash openworld_3/run_tensorrt.sh --evaluate              # full fixed test-set AP
bash openworld_3/run_tensorrt.sh --image /path/image.jpg # results/inference.json
```

The saved engine is hardware/runtime-specific (TensorRT 10.2, CUDA 11, RTX 3090). Changing GPU or TensorRT version requires rebuilding it. Outputs and CUDA Graph buffers are reused; do not concurrently call one instance. The wrapper currently initializes the source PyTorch model to reuse its exact preprocessing and postprocessing; initialization is excluded from steady-state latency.

Native ablation E2E figures include a first-use postprocessing outlier because only forward was warmed. Use the final TensorRT report for warmed E2E comparisons: both forward and postprocessing are warmed 30 times. COCO evaluation reads the original evaluator's configured `proposal_nums` (`maxDets=(1,10,100)` in this experiment); `null` size metrics mean no evaluable ground-truth objects in that size bin.
