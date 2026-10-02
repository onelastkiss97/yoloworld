#!/bin/bash
#SBATCH --job-name=yw_sp_ovd3
#SBATCH -o /home/songziqi/project/YOLO-World/newwork/openworld_1/slurm_logs/yw_sp_ovd3-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/newwork/openworld_1/slurm_logs/yw_sp_ovd3-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu4090
#SBATCH --time=01:00:00

set -euo pipefail

source /home/songziqi/anaconda3/etc/profile.d/conda.sh
conda activate YOLO-World

PY_BIN=/home/songziqi/anaconda3/envs/YOLO-World/bin/python
ROOT_DIR=/home/songziqi/project/YOLO-World
EXP_DIR=$ROOT_DIR/newwork/openworld_1
CONFIG=$EXP_DIR/sparse_zoom_ovd_openworld3_test_1280.py
CHECKPOINT=$EXP_DIR/runs/sparse_zoom_v2_m_ovd_base2_1280_e80/best_coco_bbox_mAP_epoch_80.pth
RESULT_DIR=$EXP_DIR/test_results/sparse_zoom_best80_openworld3_dynamic
TEXT_FILE=/home/songziqi/project/Data/openworld_base/texts/openworld_class_texts.json
ANNOTATION=/home/songziqi/project/Data/openworld_base/annotations/instances_test_openworld.json
CLIP_MODEL_DIR=/home/songziqi/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268

export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RESULT_DIR" "$EXP_DIR/slurm_logs"
cd "$ROOT_DIR"

test -f "$CONFIG"
test -f "$CHECKPOINT"
test -f "$TEXT_FILE"
test -f "$ANNOTATION"
test -f "$CLIP_MODEL_DIR/config.json"
test -f "$CLIP_MODEL_DIR/pytorch_model.bin"

echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $ANNOTATION"
echo "Prompts: multirotor drone | helicopter | fixed-wing aircraft"
echo "Novel class fixed-wing aircraft is prompt-only (zero-shot)."

"$PY_BIN" -u "$ROOT_DIR/newwork/tools/test_with_registered_mmyolo.py" \
  "$CONFIG" "$CHECKPOINT" \
  --work-dir "$RESULT_DIR" \
  --out "$RESULT_DIR/predictions.pkl" \
  --launcher none \
  --cfg-options test_dataloader.num_workers=8

echo "Open-world three-class evaluation finished: $RESULT_DIR"
