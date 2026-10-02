#!/bin/bash
#SBATCH --job-name=yw_sp_ovd80
#SBATCH -o /home/songziqi/project/YOLO-World/newwork/openworld_1/slurm_logs/yw_sp_ovd80-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/newwork/openworld_1/slurm_logs/yw_sp_ovd80-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu4090
#SBATCH --time=0
#SBATCH --mail-user=1755883534@qq.com

set -euo pipefail
source /home/songziqi/anaconda3/etc/profile.d/conda.sh
conda activate YOLO-World

PY_BIN=/home/songziqi/anaconda3/envs/YOLO-World/bin/python
ROOT_DIR=/home/songziqi/project/YOLO-World
EXP_DIR=$ROOT_DIR/newwork/openworld_1
DATA_ROOT=/home/songziqi/project/Data/openworld_base
CONFIG=$EXP_DIR/sparse_zoom_ovd_base2_1280_e80.py
SOURCE_WEIGHT=$ROOT_DIR/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth
INIT_WEIGHT=$EXP_DIR/weights/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft_p2_ovd.pth
CLIP_MODEL_DIR=/home/songziqi/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268

RUN_TAG=${RUN_TAG:-sparse_zoom_v2_m_ovd_base2_1280_e80}
RUN_DIR=$EXP_DIR/runs/$RUN_TAG
WORKERS=${WORKERS:-8}
BATCH=${BATCH:-2}
EPOCHS=${EPOCHS:-80}
SWITCH_EPOCH=$(( EPOCHS - 10 ))

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$WORKERS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$EXP_DIR/slurm_logs" "$EXP_DIR/weights" "$RUN_DIR"
cd "$ROOT_DIR"

"$PY_BIN" "$ROOT_DIR/openworld_1/prepare_openworld_ovd.py" --check-only
test -f "$DATA_ROOT/annotations/instances_train_base.json"
test -f "$DATA_ROOT/annotations/instances_val_base.json"
test -f "$SOURCE_WEIGHT"
test -f "$CLIP_MODEL_DIR/config.json"

if [[ ! -f "$INIT_WEIGHT" ]]; then
  "$PY_BIN" "$EXP_DIR/prepare_sparse_ovd_checkpoint.py" \
    --input "$SOURCE_WEIGHT" --output "$INIT_WEIGHT"
fi

echo "Dynamic-text base2 OVD | 1280 | epochs=$EPOCHS | batch=$BATCH"
echo "New P2/router train from scratch; pretrained P3-P5 towers are retained."

"$PY_BIN" -u "$ROOT_DIR/newwork/tools/train_with_registered_mmyolo.py" \
  "$CONFIG" --work-dir "$RUN_DIR" --launcher none \
  --cfg-options \
    train_cfg.max_epochs="$EPOCHS" \
    default_hooks.param_scheduler.max_epochs="$EPOCHS" \
    custom_hooks.2.switch_epoch="$SWITCH_EPOCH" \
    train_dataloader.batch_size="$BATCH" \
    train_dataloader.num_workers="$WORKERS" \
    optim_wrapper.optimizer.batch_size_per_gpu="$BATCH"

BEST_WEIGHT="$(find "$RUN_DIR" -maxdepth 1 -type f \
  -name 'best_coco_bbox_mAP*.pth' -print | sort | tail -n 1)"
test -n "$BEST_WEIGHT"
echo "Best dynamic checkpoint: $BEST_WEIGHT"
echo "Kept dynamic for later prompt changes and 5-shot adaptation."
