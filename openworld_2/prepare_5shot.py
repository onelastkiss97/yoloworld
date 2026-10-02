"""Freeze a nested five-shot support set without consulting test performance."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'openworld_1/fixedwing_adapt'


def main():
    support = json.loads((SOURCE/'data/support_manifest.json').read_text())['support'][:5]
    original = json.loads((SOURCE/'data/train_10shot.json').read_text())
    images = {}
    anns = {}
    for im in original['images']:
        if im['file_name'] not in images:
            images[im['file_name']] = im
            anns[im['file_name']] = [a for a in original['annotations'] if a['image_id']==im['id']]
    replay = [name for name in images if not any(a['category_id']==3 for a in anns[name])]
    assert len(replay)==40 and len(set(support))==5
    assert all(sum(a['category_id']==3 for a in anns[name])==1 for name in support)
    assert all('/train/images/' in name and Path(name).is_file() for name in support+replay)
    records = json.loads((SOURCE/'audit/records.json').read_text())
    lookup = {r['path']:r for r in records}
    heldout = {r['pixel_sha256'] for r in records if r['split']!='train'}
    assert not {lookup[n]['pixel_sha256'] for n in support+replay} & heldout
    payload = dict(info={}, images=[], annotations=[], categories=original['categories'])
    # Eight repeats of five images keep the same 40:40 novel/base sampling ratio.
    for image_id, name in enumerate(support*8+replay, 1):
        im = copy.deepcopy(images[name])
        im['id'] = image_id
        payload['images'].append(im)
        for source_ann in anns[name]:
            ann = copy.deepcopy(source_ann)
            ann.update(id=len(payload['annotations'])+1, image_id=image_id)
            payload['annotations'].append(ann)
    out = HERE/'data'
    out.mkdir(exist_ok=True)
    report = dict(novel_unique_images=5, novel_unique_instances=5, support=support,
                  selection='first five of the previously frozen 10-shot support, no test-based selection',
                  seed=20260912, support_repetitions=8, base_replay_unique_images=40,
                  validation='original val; 55 extra labeled novel images authorized',
                  test='original 938-image three-class test',
                  initialization='original base epoch100 checkpoint, NOT the 10-shot checkpoint')
    for name, content in [('train_5shot.json', json.dumps(payload)),
                          ('val.json', (SOURCE/'data/val.json').read_text()),
                          ('support_manifest.json', json.dumps(report, indent=2))]:
        target = out/name
        if target.exists() and target.read_text()!=content:
            raise RuntimeError(f'Refusing to overwrite a different manifest: {target}')
        target.write_text(content)
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
