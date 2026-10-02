"""Fixed, reproducible ten-instance support set; untouched source/test data."""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ('multirotor drone', 'helicopter', 'fixed-wing aircraft')


def coco(records):
    payload = dict(images=[], annotations=[], categories=[dict(id=i+1, name=n) for i, n in enumerate(NAMES)], info={})
    for i, r in enumerate(records, 1):
        payload['images'].append(dict(id=i, file_name=r['path'], width=r['width'], height=r['height']))
        for cls, x, y, w, h in r['boxes']:
            x1, y1 = max(0., (x-w/2)*r['width']), max(0., (y-h/2)*r['height'])
            x2, y2 = min(r['width'], (x+w/2)*r['width']), min(r['height'], (y+h/2)*r['height'])
            assert x2 > x1 and y2 > y1
            payload['annotations'].append(dict(id=len(payload['annotations'])+1,
                image_id=i, category_id=int(cls)+1, bbox=[x1, y1, x2-x1, y2-y1],
                area=(x2-x1)*(y2-y1), iscrowd=0))
    return payload


def main():
    records = json.loads((HERE/'audit/records.json').read_text())
    out = HERE / 'data'
    out.mkdir(exist_ok=True)
    rng = random.Random(20260912)
    heldout_hashes = {r['pixel_sha256'] for r in records if r['split'] != 'train'}
    novel_heldout = [r for r in records if r['split'] != 'train' and any(b[0] == 2 for b in r['boxes'])]
    candidates = [r for r in records if r['split']=='train' and sum(b[0] == 2 for b in r['boxes'])==1
                  and r['pixel_sha256'] not in heldout_hashes]
    rng.shuffle(candidates)
    support = []
    for r in candidates:
        # Conservative near-duplicate exclusion. No performance-based sample mining.
        if any(bin(r['dhash'] ^ s['dhash']).count('1') <= 3 for s in novel_heldout + support):
            continue
        support.append(r)
        if len(support) == 10:
            break
    assert len(support) == 10
    replay = []
    for cls in (0, 1):
        candidates = [r for r in records if r['split'] == 'train'
                      and {int(b[0]) for b in r['boxes']} == {cls}
                      and r['pixel_sha256'] not in heldout_hashes]
        rng.shuffle(candidates)
        replay.extend(candidates[:20])
    # Repetition changes exposure, not the number of unique novel annotations.
    train = support*4 + replay
    manifests = {'train_10shot': coco(train), 'val': coco([r for r in records if r['split']=='val'])}
    for name, payload in manifests.items():
        target = out / (name+'.json')
        content = json.dumps(payload)
        if target.exists() and target.read_text() != content:
            raise RuntimeError(f'Refusing to change frozen manifest: {target}')
        target.write_text(content)
    report = dict(seed=20260912, novel_unique_images=10, novel_unique_instances=10,
                  support=[r['path'] for r in support], base_replay_unique_images=len(replay),
                  support_repetitions=4, validation='full original val; extra novel validation labels authorized',
                  test='unchanged original 938-image mixed-class test',
                  sample_selection='seeded random among one-instance images; pixel/dHash deduplication only')
    (out/'support_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
