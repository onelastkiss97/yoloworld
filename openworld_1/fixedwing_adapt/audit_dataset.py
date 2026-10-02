"""Read-only source audit; derived reports go only in this experiment directory."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/songziqi/project/Data/openworld_base')
OUT = Path(__file__).resolve().parent / 'audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ('train', 'val', 'test'):
        for path in sorted((ROOT / split / 'images').iterdir()):
            if not path.is_file():
                continue
            label = ROOT / split / 'labels' / (path.stem + '.txt')
            if not label.exists():
                continue
            boxes = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                w, h = rgb.size
                pixel_hash = hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
                thumb = np.asarray(rgb.resize((9, 8)).convert('L'))
                dhash = int.from_bytes(np.packbits(thumb[:, 1:] > thumb[:, :-1]).tobytes(), 'big')
            records.append(dict(split=split, path=str(path), name=path.name,
                                label=str(label), width=w, height=h, boxes=boxes,
                                pixel_sha256=pixel_hash, dhash=dhash))
    by_hash = defaultdict(list)
    for r in records:
        by_hash[r['pixel_sha256']].append(r)
    overlap = [[{'split': r['split'], 'name': r['name']} for r in group]
               for group in by_hash.values() if len({r['split'] for r in group}) > 1]
    report = {'splits': {}, 'exact_cross_split_duplicates': overlap}
    novel = [r for r in records if any(b[0] == 2 for b in r['boxes'])]
    for split in ('train', 'val', 'test'):
        group = [r for r in records if r['split'] == split]
        ng = [r for r in novel if r['split'] == split]
        sizes = [b[3] * b[4] * r['width'] * r['height'] for r in ng for b in r['boxes'] if b[0] == 2]
        report['splits'][split] = dict(images=len(group),
            boxes_by_class=dict(Counter(int(b[0]) for r in group for b in r['boxes'])),
            fixedwing_images=len(ng), fixedwing_area_quantiles=np.quantile(sizes, [0, .25, .5, .75, 1]).tolist())
        sample = [ng[i] for i in np.linspace(0, len(ng)-1, min(16, len(ng)), dtype=int)]
        canvas = Image.new('RGB', (4 * 360, 4 * 270), 'white')
        for i, r in enumerate(sample):
            with Image.open(r['path']) as src:
                im = src.convert('RGB')
            draw = ImageDraw.Draw(im)
            for cls, x, y, w, h in r['boxes']:
                if cls == 2:
                    draw.rectangle(((x-w/2)*im.width, (y-h/2)*im.height,
                                    (x+w/2)*im.width, (y+h/2)*im.height), outline='red', width=3)
            im.thumbnail((354, 238))
            x, y = (i % 4)*360, (i // 4)*270
            canvas.paste(im, (x, y))
            ImageDraw.Draw(canvas).text((x+2, y+240), r['name'][:48], fill='black')
        canvas.save(OUT / f'{split}_fixedwing.jpg')
    # Candidates only: perceptual collisions are inspected, not called duplicates.
    suspect = []
    for r in novel:
        if r['split'] != 'train':
            continue
        for s in novel:
            if s['split'] == 'train':
                continue
            distance = bin(r['dhash'] ^ s['dhash']).count('1')
            if distance <= 3:
                suspect.append(dict(train=r['name'], other=s['name'], split=s['split'], dhash_distance=distance))
    report['novel_perceptual_overlap_candidates'] = suspect
    (OUT / 'records.json').write_text(json.dumps(records))
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
