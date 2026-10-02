#!/usr/bin/env python3
"""Reparameterize a trained dynamic Sparse-Zoom YOLO-World checkpoint."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HEAD_RE = re.compile(
    r'^(bbox_head\.head_module\.cls_contrasts\.\d+)\.logit_scale$')
GUIDE_RE = re.compile(r'^(neck\..+\.attn_block)\.guide_fc\.weight$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Dynamic checkpoint')
    parser.add_argument('--text-embed', required=True, help='N x 512 .npy')
    parser.add_argument('--output', required=True, help='Output rep checkpoint')
    return parser.parse_args()


def convert_head(state_dict, embeds):
    prefixes = sorted(
        match.group(1) for key in state_dict
        if (match := HEAD_RE.match(key)) is not None)
    for prefix in prefixes:
        scale = state_dict.pop(prefix + '.logit_scale')
        bias = state_dict.pop(prefix + '.bias')
        state_dict[prefix + '.conv.weight'] = (
            embeds * scale.exp()).to(scale).unsqueeze(-1).unsqueeze(-1)
        state_dict[prefix + '.conv.bias'] = bias.expand(embeds.shape[0]).clone()
    return prefixes


def convert_neck(state_dict, embeds):
    prefixes = sorted(
        match.group(1) for key in state_dict
        if (match := GUIDE_RE.match(key)) is not None)
    for prefix in prefixes:
        weight = state_dict.pop(prefix + '.guide_fc.weight')
        bias = state_dict.pop(prefix + '.guide_fc.bias')
        guide = embeds.to(weight) @ weight.t() + bias
        num_heads = state_dict[prefix + '.bias'].numel()
        chunks = guide.chunk(num_heads, dim=1)
        for index, chunk in enumerate(chunks):
            state_dict[f'{prefix}.guide_convs.{index}.weight'] = (
                chunk.unsqueeze(-1).unsqueeze(-1))
    return prefixes


def convert_router(state_dict, embeds):
    prefix = 'neck.router'
    weight_key = prefix + '.text_proj.weight'
    if weight_key not in state_dict:
        raise KeyError('Dynamic neck.router.text_proj was not found.')
    weight = state_dict.pop(weight_key)
    bias = state_dict.pop(prefix + '.text_proj.bias')
    projected = embeds.to(weight) @ weight.t() + bias
    projected = F.normalize(projected, dim=-1, eps=1e-6)
    state_dict[prefix + '.guide_conv.weight'] = projected[:, :, None, None]


def main():
    args = parse_args()
    checkpoint = torch.load(args.model, map_location='cpu')
    state_dict = checkpoint['state_dict']
    embeds = torch.from_numpy(np.load(args.text_embed)).float()
    if embeds.ndim == 3 and embeds.shape[0] == 1:
        embeds = embeds[0]
    if embeds.ndim != 2:
        raise ValueError(f'Expected N x D embeddings, got {tuple(embeds.shape)}')

    # Text encoder/prompt parameters do not exist in the rep model.
    for key in list(state_dict):
        if 'text_model' in key or key == 'embeddings' or key.startswith('adapter.'):
            state_dict.pop(key)
    heads = convert_head(state_dict, embeds)
    neck_blocks = convert_neck(state_dict, embeds)
    convert_router(state_dict, embeds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f'Reparameterized {len(heads)} head levels, '
          f'{len(neck_blocks)} PAN attention blocks, and the P2 router.')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()

