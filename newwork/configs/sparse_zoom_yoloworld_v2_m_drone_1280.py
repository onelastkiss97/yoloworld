"""Dynamic-text Sparse-Zoom YOLO-World V2-M on UAV/RGB + drone-yolo-rgb."""

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_m_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world', 'newwork'], allow_failed_imports=False)

data_root = '/home/songziqi/project/'
ann_root = ('/home/songziqi/project/YOLO-World/'
            'drone-yolo-rgb+UAV1/annotations')
classes = ('drone', )
metainfo = dict(classes=classes)

num_classes = 1
max_epochs = 80
close_mosaic_epochs = 10
save_epoch_intervals = 5
img_scale = (1280, 1280)
deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768
text_channels = 512
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 2
persistent_workers = False
mixup_prob = 0.1
strides = [4, 8, 16, 32]

load_from = (
    '/home/songziqi/project/YOLO-World/newwork/weights/'
    'yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft_p2_dynamic.pth')

model = dict(
    _delete_=True,
    type='SparseZoomYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_classes,
    num_test_classes=num_classes,
    prompt_dim=text_channels,
    num_prompts=num_classes,
    embedding_path=(
        '/home/songziqi/project/YOLO-World/drone-yolo-rgb+UAV1/'
        'texts/drone_clip_vit_b32.npy'),
    reparameterized=False,
    freeze_prompt=True,
    router_loss_weight=1.0,
    router_coverage_weight=0.5,
    sparse_budget_weight=0.1,
    small_object_max_size=96.0,
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='MultiModalYOLOBackbone',
        text_model=None,
        with_text_model=False,
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=last_stage_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            out_indices=(1, 2, 3, 4),
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True))),
    neck=dict(
        type='TextGuidedSparseYOLOWorldPAFPN',
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        p2_in_channels=128,
        p2_out_channels=128,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        num_csp_blocks=3,
        guide_channels=text_channels,
        embed_channels=[128, 256, last_stage_out_channels // 2],
        num_heads=[4, 8, last_stage_out_channels // 2 // 32],
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        text_channels=text_channels,
        router_channels=64,
        reparameterized=False,
        window_size=8,
        sparse_ratio=0.25,
        refinement_blocks=2),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            num_classes=num_classes,
            in_channels=[128, 256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            use_bn_head=True,
            embed_dims=text_channels,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=1.5 / 4)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=False,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

final_transform = [
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]
mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(0.1, 1.9),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]
last_augment = [
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5)
]
train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_augment,
    *final_transform
]
train_pipeline_stage2 = [
    *pre_transform,
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
        scaling_ratio_range=(0.1, 1.9),
        max_aspect_ratio=100.,
        border_val=(114, 114, 114)),
    *last_augment,
    *final_transform
]

train_dataset = dict(
    type='YOLOv5CocoDataset',
    data_root=data_root,
    ann_file=f'{ann_root}/instances_train.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    metainfo=metainfo,
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=persistent_workers,
    collate_fn=dict(type='yolow_collate'),
    dataset=train_dataset)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114.0)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
val_dataset = dict(
    type='YOLOv5CocoDataset',
    data_root=data_root,
    ann_file=f'{ann_root}/instances_val.json',
    data_prefix=dict(img=''),
    test_mode=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    metainfo=metainfo,
    pipeline=test_pipeline)
test_dataset = dict(
    type='YOLOv5CocoDataset',
    data_root=data_root,
    ann_file=f'{ann_root}/instances_test.json',
    data_prefix=dict(img=''),
    test_mode=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    metainfo=metainfo,
    pipeline=test_pipeline)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    constructor='YOLOWv5OptimizerConstructor',
    paramwise_cfg=dict(
        custom_keys={
            'logit_scale': dict(weight_decay=0.0),
            'embeddings': dict(weight_decay=0.0)
        }))
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear', lr_factor=0.01, max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best='coco/bbox_mAP',
        rule='greater',
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=f'{ann_root}/instances_val.json',
    metric='bbox')
test_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=f'{ann_root}/instances_test.json',
    metric='bbox')

