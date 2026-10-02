#!/bin/bash
#SBATCH --job-name=yw_sparse_test
#SBATCH -o /home/songziqi/project/YOLO-World/newwork/slurm_logs/yw_sparse_test-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/newwork/slurm_logs/yw_sparse_test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu4090
#SBATCH --time=12:00:00

set -euo pipefail

source /home/songziqi/anaconda3/etc/profile.d/conda.sh
conda activate YOLO-World

PY_BIN="/home/songziqi/anaconda3/envs/YOLO-World/bin/python"
ROOT_DIR="/home/songziqi/project/YOLO-World"
EXP_DIR="$ROOT_DIR/newwork"
CONFIG="$EXP_DIR/configs/sparse_zoom_yoloworld_v2_m_drone_1280_rep.py"
TEXT_EMBED="$ROOT_DIR/drone-yolo-rgb+UAV1/texts/drone_clip_vit_b32.npy"

: "${CHECKPOINT:?Submit with CHECKPOINT=/absolute/path/to/best.pth sbatch ...}"
REP_CHECKPOINT="${REP_CHECKPOINT:-$EXP_DIR/weights/$(basename "${CHECKPOINT%.pth}")_rep.pth}"

export OMP_NUM_THREADS="${WORKERS:-8}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
mkdir -p "$EXP_DIR/slurm_logs" "$EXP_DIR/weights" "$EXP_DIR/test_results"
cd "$ROOT_DIR"

"$PY_BIN" "$EXP_DIR/tools/reparameterize_sparse_yoloworld.py" \
  --model "$CHECKPOINT" \
  --text-embed "$TEXT_EMBED" \
  --output "$REP_CHECKPOINT"

"$PY_BIN" -u "$EXP_DIR/tools/test_with_registered_mmyolo.py" \
  "$CONFIG" "$REP_CHECKPOINT" \
  --work-dir "$EXP_DIR/test_results/$(basename "${REP_CHECKPOINT%.pth}")" \
  --launcher none

echo "Reparameterized checkpoint: $REP_CHECKPOINT"

