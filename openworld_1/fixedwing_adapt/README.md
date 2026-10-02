# Fixed-wing aircraft 10-shot adaptation

Quick access from `openworld_1/`: `01_baseline.sh`, `02_train_10shot.sh`, and `03_evaluate.sh` are symlinks to this experiment's three Slurm scripts. The current jobs are already submitted; these links are for navigation and future runs. `evaluate.sh` takes a completed training run directory as its argument.

At the user's request, obsolete base-training checkpoints/logs, reparameterized weights/test outputs, old preparation/training scripts, and caches were removed. The required epoch-100 best checkpoint and imported base configuration/helpers remain at their original paths. Dataset audits and the frozen support manifest are retained for this experiment's reproducibility.

## Objective and agreed protocol

Raise **fixed-wing aircraft class AP50** from historical **0.395** to **>=0.55**, evaluated on the unchanged original **938-image, three-class test set**. This is distinct from the unrelated UAV RGB `0-shots`/`5-shots` experiments. The historical three-class mean AP50 is 0.767, not the target metric.

The user authorized ten shots from original `train`, original `val` for model selection, and **4090 GPUs only**. Ten distinct images containing exactly ten novel instances are frozen in `data/support_manifest.json`, seed 20260912. Validation uses 55 additional labeled novel images and 1,461 base-only images; this is **10-shot training with additional novel validation supervision**, not a ten-total-label experiment. Test is never included in fine-tuning or checkpoint selection.

## Audited original training logic

`../yolo_world_v2_m_openworld_standard_1280.py` starts from the public YOLO-World M 1280 pretraining weight and trains on 6,269 base-only images (4,405 multirotor boxes, 2,220 helicopter boxes). CLIP is frozen. Base-only validation selects the best checkpoint by bbox AP50:95; the selected checkpoint is epoch 100. The training shell then reparameterizes three text prompts and evaluates the three-class test set. Novel AP50 is 0.395, novel AP50:95 is 0.179. This is zero-shot with respect to the local novel training split; the public pretraining data have not been audited for aircraft exposure.

## Dataset findings

| Original split | Images | Fixed-wing images | Fixed-wing boxes |
|---|---:|---:|---:|
| train | 6,712 | 443 | 467 |
| val | 1,516 | 55 | 55 |
| test | 938 | 56 | 56 |

`audit/report.json` contains pixel SHA256 overlap findings and perceptual-hash candidates. There are **463 cross-split exact duplicate groups**, involving base classes; no exact cross-split fixed-wing duplicate was found. Two novel train/heldout dHash candidates were found, and support sampling conservatively excludes close dHash matches against heldout novel images and other selected supports. This does not prove that different frames from the same source video are independent. Existing base-model contamination is not undone by fine-tuning and limits interpretation of base metrics.

`audit/*_fixedwing.jpg` provides image/box contact sheets. The sample images include large, elongated aircraft, RC models, runway/sky scenes, and apparent rotorcraft under the fixed-wing label. Original annotations remain unchanged; potentially ambiguous labels should be reviewed separately. No test-label correction is used to meet the target.

## Method and GitHub references

1. [TFA / FsDet official implementation](https://github.com/ucbdrive/few-shot-object-detection): two-stage few-shot adaptation, freezing much of the model and balancing novel/base examples. Our first trial adapts this principle to YOLO-World by freezing encoders, tuning neck/head, and replaying base samples. It is **not a literal reproduction of TFA**.
2. [YOLO-World prompt tuning](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/prompt_yolo_world.md): learned text/image embeddings and optional adapter are available in the existing repository. A relevant follow-up if initial fitting is insufficient; not yet implemented in this experiment.
3. [DeFRCN](https://github.com/er-muyue/DeFRCN) and [DCFS](https://github.com/gaobb/DCFS): official few-shot/generalized few-shot detection implementations. Their Faster R-CNN/Detectron2 architecture is not a drop-in YOLO-World module; no external implementation has been installed or claimed to improve this dataset.

The first experiment uses ten novel images repeated four times plus twenty existing training images per base class (40 unique base replay images); encoder weights and all BN running statistics are frozen; AdamW LR 3e-5; mild affine/HSV/flip augmentation; 1280 resolution; batch 4; 600 iterations; validation every 100 iterations. Save-best explicitly uses `coco/fixedwing_ap50`. Each checkpoint is selected solely on validation.

## Files and execution

- `baseline.sh`: original multimodal checkpoint evaluation, saved predictions for comparison with historical reparameterized output.
- `train.sh`: 10-shot fitting; no existing base checkpoint is overwritten.
- `evaluate.sh RUN_DIR`: held-out test of the single validation-selected checkpoint.
- `slurm_logs/`: all current experiment Slurm `.out/.err` files.
- `runs/`: weights, expanded configs, and training/validation metrics.
- `test_results/`: predictions and `summary.json`, including `target_met` determined from the **unrounded** AP50.
- `verify_setup.py`: CPU checks of support count/isolation, dataset transforms, and AP50 on perfect/non-overlapping synthetic detections.

Submitted 4090 jobs: baseline **85511**, training **85512**, automatic held-out test **85513** (depends on successful completion of 85512). Prior 3090 jobs **85504/85505** were cancelled after the user's 4090-only instruction. The partial baseline 85504 is not a completed result.

The cancelled 85504 job's partial logs and config snapshots were deleted at the user's request; no completed evaluation or trained weight was removed.

CPU verification passed: ten distinct novel instances, pixel-level support/replay isolation from held-out images, dataset transform/category mapping, and AP50 extraction for perfect and non-overlapping predictions. GPU training/loss checks await allocation.

**Status: experiments queued; no new AP50 result and no claim that 55% has been achieved.**
