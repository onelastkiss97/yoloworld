#!/bin/bash
#SBATCH --job-name=yw_sparse_p2
#SBATCH -o /home/songziqi/project/YOLO-World/newwork/slurm_logs/yw_sparse_p2-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/newwork/slurm_logs/yw_sparse_p2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu4090
#SBATCH --time=48:00:00
#SBATCH --mail-user=1755883534@qq.com

set -euo pipefail

source /home/songziqi/anaconda3/etc/profile.d/conda.sh
conda activate YOLO-World

PY_BIN="/home/songziqi/anaconda3/envs/YOLO-World/bin/python"
ROOT_DIR="/home/songziqi/project/YOLO-World"
EXP_DIR="$ROOT_DIR/newwork"
CONFIG="$EXP_DIR/configs/sparse_zoom_yoloworld_v2_m_drone_1280.py"
SOURCE_WEIGHT="$ROOT_DIR/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth"
P2_WEIGHT="$EXP_DIR/weights/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft_p2_dynamic.pth"

RUN_TAG="${RUN_TAG:-sparse_zoom_v2_m_drone_1280_e80}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-10}"
SPARSE_RATIO="${SPARSE_RATIO:-0.25}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
SWITCH_EPOCH=$(( EPOCHS - CLOSE_MOSAIC ))
if (( SWITCH_EPOCH < 0 )); then
  SWITCH_EPOCH=0
fi

export OMP_NUM_THREADS="$WORKERS"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p "$EXP_DIR/slurm_logs" "$EXP_DIR/runs" "$EXP_DIR/weights"
cd "$ROOT_DIR"

[[ -f "$SOURCE_WEIGHT" ]] || {
  echo "Missing source checkpoint: $SOURCE_WEIGHT" >&2
  exit 1
}

if [[ ! -f "$P2_WEIGHT" ]]; then
  "$PY_BIN" "$EXP_DIR/tools/prepare_p2_checkpoint.py" \
    --input "$SOURCE_WEIGHT" \
    --output "$P2_WEIGHT"
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "P2 window=$WINDOW_SIZE, sparse ratio=$SPARSE_RATIO, batch=$BATCH"

"$PY_BIN" -u "$EXP_DIR/tools/train_with_registered_mmyolo.py" "$CONFIG" \
  --work-dir "$EXP_DIR/runs/$RUN_TAG" \
  --launcher none \
  --cfg-options \
    train_cfg.max_epochs="$EPOCHS" \
    default_hooks.param_scheduler.max_epochs="$EPOCHS" \
    custom_hooks.1.switch_epoch="$SWITCH_EPOCH" \
    train_dataloader.batch_size="$BATCH" \
    train_dataloader.num_workers="$WORKERS" \
    val_dataloader.num_workers="$WORKERS" \
    test_dataloader.num_workers="$WORKERS" \
    model.neck.sparse_ratio="$SPARSE_RATIO" \
    model.neck.window_size="$WINDOW_SIZE"

echo "Training finished: $EXP_DIR/runs/$RUN_TAG"

