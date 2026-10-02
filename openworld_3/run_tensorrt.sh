#!/usr/bin/env bash
set -euo pipefail
cd /home/songziqi/project/YOLO-World
export PYTHONPATH="/home/songziqi/project/YOLO-World/openworld_3/vendor:/home/songziqi/project/YOLO-World${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/songziqi/anaconda3/envs/YOLO-World/bin/python -u openworld_3/tensorrt_experiment.py "$@"
