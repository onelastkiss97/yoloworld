#!/bin/bash
#SBATCH --job-name=fw_10shot_test
#SBATCH -o /home/songziqi/project/YOLO-World/openworld_1/fixedwing_adapt/slurm_logs/fw_test-%j.out
#SBATCH -e /home/songziqi/project/YOLO-World/openworld_1/fixedwing_adapt/slurm_logs/fw_test-%j.err
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
python -u openworld_1/fixedwing_adapt/evaluate_best.py "$1"
