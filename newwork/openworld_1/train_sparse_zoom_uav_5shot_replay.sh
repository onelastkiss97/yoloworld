#!/bin/bash
#SBATCH --job-name=yw_sp_uav5
#SBATCH -o /home/songziqi/project/YOLO-World/newwork/openworld_1/5-shots/slurm_logs/yw_sp_uav5-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/newwork/openworld_1/5-shots/slurm_logs/yw_sp_uav5-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu4090
#SBATCH --time=02:00:00

set -euo pipefail

source /home/songziqi/anaconda3/etc/profile.d/conda.sh
conda activate YOLO-World

PY_BIN=/home/songziqi/anaconda3/envs/YOLO-World/bin/python
ROOT_DIR=/home/songziqi/project/YOLO-World
EXP_DIR=$ROOT_DIR/newwork/openworld_1
FIVE_DIR=$EXP_DIR/5-shots
DATA_ROOT=/home/songziqi/project/Data/UAV/5-shots
CONFIG=$EXP_DIR/sparse_zoom_uav_5shot_replay_1000iter.py
SOURCE_WEIGHT=$EXP_DIR/runs/sparse_zoom_v2_m_ovd_base2_1280_e80/best_coco_bbox_mAP_epoch_80.pth
RUN_DIR=$FIVE_DIR/runs/sparse_zoom_uav_5shot_replay_1000iter
ZERO_DIR=$FIVE_DIR/zero_shot/sparse_zoom_best80_small_drone
TEST_DIR=$FIVE_DIR/test_results/sparse_zoom_uav_5shot_replay_best_map50
CLIP_MODEL_DIR=/home/songziqi/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
MAX_ITERS=${MAX_ITERS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-100}
BATCH=${BATCH:-4}

export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$FIVE_DIR/slurm_logs" "$RUN_DIR" "$ZERO_DIR" "$TEST_DIR"
cd "$ROOT_DIR"

test -f "$CONFIG"
test -f "$SOURCE_WEIGHT"
test -f "$DATA_ROOT/five_shot_replay_audit.json"
test -f "$DATA_ROOT/annotations/instances_train_5shot_replay.json"
test -f "$DATA_ROOT/annotations/instances_val_small_drone.json"
test -f "$DATA_ROOT/annotations/instances_test_small_drone.json"
test -f "$DATA_ROOT/texts/train_class_texts.json"
test -f "$DATA_ROOT/texts/eval_class_texts.json"
test -f "$CLIP_MODEL_DIR/config.json"
test -f "$CLIP_MODEL_DIR/pytorch_model.bin"

if [[ -e "$RUN_DIR/last_checkpoint" ]]; then
  echo "Refusing to overwrite an existing completed/interrupted run: $RUN_DIR" >&2
  exit 2
fi

echo "Sparse-Zoom zero-shot baseline on held-out UAV/RGB test"
"$PY_BIN" -u "$ROOT_DIR/newwork/tools/test_with_registered_mmyolo.py" \
  "$CONFIG" "$SOURCE_WEIGHT" \
  --work-dir "$ZERO_DIR" --out "$ZERO_DIR/predictions.pkl" --launcher none

echo "Five-shot training: $MAX_ITERS iterations, validate every $VAL_INTERVAL"
echo "Prompts: small drone | multirotor drone | helicopter"
echo "LR: 1e-5 | image size: 1280 | batch: $BATCH"
echo "Frozen: CLIP + image backbone | trainable: sparse neck/router + P2-P5 head"

"$PY_BIN" -u "$ROOT_DIR/newwork/tools/train_with_registered_mmyolo.py" \
  "$CONFIG" --work-dir "$RUN_DIR" --launcher none \
  --cfg-options train_cfg.max_iters="$MAX_ITERS" \
  train_cfg.val_begin="$VAL_INTERVAL" train_cfg.val_interval="$VAL_INTERVAL" \
  default_hooks.checkpoint.interval="$VAL_INTERVAL" \
  train_dataloader.batch_size="$BATCH"

BEST_WEIGHT="$(find "$RUN_DIR" -maxdepth 1 -type f \
  -name 'best_coco_bbox_mAP_50_iter_*.pth' -print | sort -V | tail -n 1)"
test -n "$BEST_WEIGHT"
echo "Best validation mAP50 checkpoint: $BEST_WEIGHT"

"$PY_BIN" -u "$ROOT_DIR/newwork/tools/test_with_registered_mmyolo.py" \
  "$CONFIG" "$BEST_WEIGHT" \
  --work-dir "$TEST_DIR" --out "$TEST_DIR/predictions.pkl" --launcher none

echo "Sparse-Zoom five-shot training and held-out test completed: $TEST_DIR"
