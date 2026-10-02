#!/usr/bin/env python3
"""Shift a released three-level YOLO-World head to P2--P5 indexing.

P3/P4/P5 head parameters are moved from indices 0/1/2 to 1/2/3.  Index 0
is intentionally absent so that MMEngine initializes the new P2 head.
Backbone and neck names are left untouched.
"""

import argparse
import re
from pathlib import Path

import torch


HEAD_RE = re.compile(
    r'^(bbox_head\.head_module\.(?:cls_preds|reg_preds|cls_contrasts))\.(\d+)(\..+)$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Released checkpoint')
    parser.add_argument('--output', required=True, help='Migrated checkpoint')
    return parser.parse_args()


def migrate(state_dict):
    migrated = {}
    moved = 0
    for key, value in state_dict.items():
        match = HEAD_RE.match(key)
        if match:
            key = f'{match.group(1)}.{int(match.group(2)) + 1}{match.group(3)}'
            moved += 1
        migrated[key] = value
    return migrated, moved


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict, moved = migrate(state_dict)
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = state_dict
        output = checkpoint
    else:
        output = state_dict
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f'Moved {moved} P3--P5 head tensors by +1 level.')
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()

