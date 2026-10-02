"""Sparse-Zoom UAV five-shot adaptation with 15-image base replay.

This follows the data boundary and model-selection protocol in HANDOFF.md.
Only the dynamic neck/router and four-level detection head are optimized.
"""

_base_ = './sparse_zoom_ovd_base2_1280_e80.py'

custom_imports = dict(
    imports=[
        'yolo_world', 'newwork', 'openworld_1.standard_ovd_components'
    ],
    allow_failed_imports=False)

data_root = '/home/songziqi/project/Data/UAV/5-shots'
ann_root = f'{data_root}/annotations'
text_root = f'{data_root}/texts'
classes = ('small drone', 'multirotor drone', 'helicopter')
train_metainfo = dict(classes=classes)
eval_classes = ('small drone', )
eval_metainfo = dict(classes=eval_classes)
num_train_classes = 3
num_test_classes = 1
max_iters = 1000
val_interval = 100
base_lr = 1e-5
weight_decay = 0.01
batch_size = 4
img_scale = (1280, 1280)

load_from = (
    '/home/songziqi/project/YOLO-World/newwork/openworld_1/runs/'
    'sparse_zoom_v2_m_ovd_base2_1280_e80/'
    'best_coco_bbox_mAP_epoch_80.pth')

model = dict(
    type='FrozenBackboneTextGuidedSparseOVDDetector',
    num_train_classes=num_train_classes,
    num_test_classes=num_test_classes,
    backbone=dict(
        image_model=dict(frozen_stages=4, norm_eval=True)),
    bbox_head=dict(
        head_module=dict(num_classes=num_train_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_train_classes)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(0.9, 1.1),
        max_aspect_ratio=100.,
        border_val=(114, 114, 114)),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='RandomLoadText',
        num_neg_samples=(2, 2),
        max_num_samples=num_train_classes,
        padding_to_max=True,
        padding_value=''),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
            'flip_direction', 'texts')),
]

train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file=f'{ann_root}/instances_train_5shot_replay.json',
        data_prefix=dict(img='mixed_train/images/'),
        metainfo=train_metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=f'{text_root}/train_class_texts.json',
    pipeline=train_pipeline)
train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset)

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


def eval_dataset(split):
    return dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='/home/songziqi/project/Data/UAV/RGB',
            ann_file=f'{ann_root}/instances_{split}_small_drone.json',
            data_prefix=dict(img=f'images/{split}/'),
            metainfo=eval_metainfo,
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=f'{text_root}/eval_class_texts.json',
        pipeline=test_pipeline)


val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=eval_dataset('val'))
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=eval_dataset('test'))

val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    classwise=True,
    ann_file=f'{ann_root}/instances_val_small_drone.json',
    metric='bbox',
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    classwise=True,
    ann_file=f'{ann_root}/instances_test_small_drone.json',
    metric='bbox',
    proposal_nums=(100, 1, 10))

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_begin=val_interval,
    val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Preserve the original five-shot optimization protocol.
param_scheduler = None
default_hooks = dict(
    param_scheduler=None,
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval,
        filename_tmpl='iter_{}.pth',
        max_keep_ckpts=-1,
        save_best='coco/bbox_mAP_50',
        rule='greater'))
custom_hooks = [
    dict(
        type='TrainableParameterAuditHook',
        allowed_prefixes=('neck.', 'bbox_head.')),
    dict(type='PositiveDetectionLossHook', check_iters=25),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0, norm_type=2),
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    constructor='YOLOWv5OptimizerConstructor')

log_processor = dict(by_epoch=False, window_size=20)
work_dir = (
    '/home/songziqi/project/YOLO-World/newwork/openworld_1/5-shots/runs/'
    'sparse_zoom_uav_5shot_replay_1000iter')

