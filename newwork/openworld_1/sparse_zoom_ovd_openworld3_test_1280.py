"""Dynamic three-class open-vocabulary test for Sparse-Zoom YOLO-World.

The detector was trained only on the two base classes.  The third class is
introduced here exclusively through the text prompt, matching HANDOFF.md.
"""

_base_ = './sparse_zoom_ovd_base2_1280_e80.py'

data_root = '/home/songziqi/project/Data/openworld_base'
ann_file = f'{data_root}/annotations/instances_test_openworld.json'
text_file = f'{data_root}/texts/openworld_class_texts.json'
img_scale = (1280, 1280)

# Keep this order identical to the COCO annotations and the original test.
classes = (
    'multirotor drone',
    'helicopter',
    'fixed-wing aircraft',
)
metainfo = dict(classes=classes)

# The head remains dynamic: the actual number of logits is inferred from the
# three text features at prediction time.  No reparameterization is performed.
model = dict(num_test_classes=3)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'pad_param', 'texts')),
]

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=dict(img='test/images/'),
            metainfo=metainfo,
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=text_file,
        pipeline=test_pipeline))

test_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    ann_file=ann_file,
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 1, 10))

work_dir = (
    '/home/songziqi/project/YOLO-World/newwork/openworld_1/'
    'test_results/sparse_zoom_best80_openworld3_dynamic')

