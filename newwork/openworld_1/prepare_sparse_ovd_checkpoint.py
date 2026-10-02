#!/usr/bin/env python3
"""Safely migrate a released three-level checkpoint to P2--P5 indexing."""

import argparse
import re
from pathlib import Path

import torch


HEAD_RE = re.compile(
    r'^(bbox_head\.head_module\.(?:cls_preds|reg_preds|cls_contrasts))'
    r'\.(\d+)(\..+)$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def migrate(state_dict):
    if any(HEAD_RE.match(key) and int(HEAD_RE.match(key).group(2)) >= 3
           for key in state_dict):
        raise ValueError('Checkpoint already contains a fourth head level.')
    output = {}
    moved = 0
    for key, value in state_dict.items():
        match = HEAD_RE.match(key)
        if match:
            key = (f'{match.group(1)}.{int(match.group(2)) + 1}'
                   f'{match.group(3)}')
            moved += 1
        output[key] = value
    return output, moved


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    checkpoint['state_dict'], moved = migrate(checkpoint['state_dict'])
    # This field is only used when resuming, but migrate it too so the output
    # is safe for either load_from or resume workflows.
    if 'ema_state_dict' in checkpoint:
        checkpoint['ema_state_dict'], _ = migrate(
            checkpoint['ema_state_dict'])
    checkpoint.setdefault('meta', {})['sparse_p2_initialization'] = dict(
        source=str(Path(args.input).resolve()),
        rule='P3/P4/P5 head indices 0/1/2 -> 1/2/3; P2 is new')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f'Migrated {moved} head tensors; P2/router remain uninitialized.')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()

