# Fixed-wing 5-shot experiment

Submitted job: **85544**, partition **gpu4090**. Training output directory: `runs/neck_5shot_85544`; final test output: `test_results/neck_5shot_85544`. Submission-time checks passed for base checkpoint/model/test equivalence and the five-shot data pipeline.

Run `sbatch openworld_2/train_5shot.sh` from the YOLO-World repository. The single 4090 job trains, selects its best checkpoint on the original validation set, then tests it on the unchanged original 938-image mixed-class test set.

- Support: first five images from the frozen 10-shot support, exactly five fixed-wing instances. No outcome-based resampling.
- Initialization: original base-only epoch100 best checkpoint, **not the 10-shot fine-tuned weight**.
- Replay: same 40 unique base images. Five novel images repeat eight times, retaining the previous experiment's 40:40 novel/base exposure ratio and 80-entry training dataset.
- Same architecture, frozen encoders/BN statistics, mild augmentations, AdamW LR 3e-5, batch 4, 1280 resolution, 600 iterations, validation every 100 iterations.
- Selection: `coco/fixedwing_ap50` on original val, including 55 additional labeled novel images as previously authorized. Results are five-shot training with additional validation supervision.
- `slurm_logs/`: training and final test output/error logs.
- `runs/neck_5shot_JOBID/`: weights, expanded configuration, train/validation logs.
- `test_results/neck_5shot_JOBID/`: final per-class metrics, predictions.pkl, summary.json.
- `data/support_manifest.json`: the exact five selected image paths and protocol.

The configuration inherits the existing implementation and base checkpoint from `openworld_1`; do not delete those dependencies. New outputs are confined to `openworld_2`. Base-model dataset overlap and possible source-frame/label issues documented in the previous experiment still apply. No new results are claimed before the job completes.
