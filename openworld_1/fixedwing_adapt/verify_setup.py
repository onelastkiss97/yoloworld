"""CPU checks for data isolation, category mapping, and AP50 extraction."""
import json
import tempfile
from pathlib import Path

import numpy as np
from mmengine import Config
from mmyolo.utils import register_all_modules
from mmyolo.registry import DATASETS

from openworld_1.fixedwing_adapt.components import FixedWingCocoMetric

HERE = Path(__file__).resolve().parent


def main():
    register_all_modules()
    cfg = Config.fromfile(str(HERE / 'finetune.py'))
    data = json.loads((HERE / 'data/train_10shot.json').read_text())
    byid = {r['id']: r for r in data['images']}
    novel_files = {byid[a['image_id']]['file_name'] for a in data['annotations'] if a['category_id'] == 3}
    assert len(novel_files) == 10
    assert len([a for a in data['annotations'] if a['category_id']==3]) == 40
    assert all('/train/images/' in r['file_name'] for r in data['images'])
    records = json.loads((HERE/'audit/records.json').read_text())
    lookup = {r['path']: r for r in records}
    train_hashes = {lookup[r['file_name']]['pixel_sha256'] for r in data['images']}
    heldout_hashes = {r['pixel_sha256'] for r in records if r['split'] != 'train'}
    assert not train_hashes & heldout_hashes
    ds = DATASETS.build(cfg.train_dataloader.dataset)
    sample = ds[0]
    assert sample['data_samples'].gt_instances.labels.numel() > 0
    labels = sample['data_samples'].gt_instances.labels.tolist()
    assert all(0 <= label < 3 for label in labels)
    assert len(sample['data_samples'].texts) == 3
    assert cfg.test_dataloader.dataset.dataset.ann_file.endswith('instances_test_openworld.json')
    # A perfect synthetic prediction should yield 1, a wrong-category prediction 0.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'gt.json'
        path.write_text(json.dumps(dict(info={}, images=[dict(id=1,width=100,height=100,file_name='unused.jpg')],
            categories=[dict(id=3,name='fixed-wing aircraft')],
            annotations=[dict(id=1,image_id=1,category_id=3,bbox=[10,10,20,20],area=400,iscrowd=0)])))
        for box, expected in (([10,10,30,30], 1.), ([60,60,80,80], 0.)):
            metric = FixedWingCocoMetric(ann_file=str(path), proposal_nums=(100,1,10))
            metric.dataset_meta = dict(classes=('fixed-wing aircraft',))
            result = metric.compute_metrics([(dict(), dict(img_id=1,bboxes=np.array([box]),
                                                        labels=np.array([0]),scores=np.array([.9])))])
            assert abs(result['fixedwing_ap50'] - expected) < 1e-6, result
    print('PASS: 10 distinct novel instances; train/heldout pixel isolation; dataset transform; exact AP50 extraction.')


if __name__ == '__main__':
    main()
