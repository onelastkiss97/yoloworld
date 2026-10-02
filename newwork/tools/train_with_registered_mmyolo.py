#!/usr/bin/env python3
"""Register MMYOLO/newwork modules, then delegate to tools/train.py."""

import runpy
import sys
from pathlib import Path

from mmyolo.utils import register_all_modules


register_all_modules()
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
import newwork  # noqa: E402,F401

sys.argv[0] = str(root_dir / 'tools' / 'train.py')
runpy.run_path(sys.argv[0], run_name='__main__')

