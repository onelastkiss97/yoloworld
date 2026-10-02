#!/bin/bash
#SBATCH --job-name=fw_baseline
#SBATCH -o /home/songziqi/project/YOLO-World/openworld_1/fixedwing_adapt/slurm_logs/fw_baseline-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/openworld_1/fixedwing_adapt/slurm_logs/fw_baseline-%j.err
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
cd /home/songziqi/project/YOLO-World
export PYTHONPATH="/home/songziqi/project/YOLO-World:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
RUN_DIR="openworld_1/fixedwing_adapt/runs/baseline_${SLURM_JOB_ID}"
python -u openworld_1/test_with_registered_mmyolo.py \
  openworld_1/fixedwing_adapt/baseline.py \
  openworld_1/runs/v2_m_ovd_standard_base2_novel_fixedwing_1280_e150/best_coco_bbox_mAP_epoch_100.pth \
  --work-dir "$RUN_DIR" --out "$RUN_DIR/predictions.pkl"
