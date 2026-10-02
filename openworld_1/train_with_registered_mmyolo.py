#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path
from mmyolo.utils import register_all_modules

register_all_modules()
root = Path(__file__).resolve().parents[1]
sys.argv[0] = str(root / 'tools' / 'train.py')
runpy.run_path(sys.argv[0], run_name='__main__')
