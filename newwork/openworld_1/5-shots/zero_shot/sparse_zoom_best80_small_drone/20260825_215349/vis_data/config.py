_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.9
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
ann_root = '/home/songziqi/project/Data/UAV/5-shots/annotations'
backend_args = None
base_lr = 1e-05
batch_shapes_cfg = None
batch_size = 4
classes = (
    'small drone',
    'multirotor drone',
    'helicopter',
)
close_mosaic_epochs = 10
custom_hooks = [
    dict(
        allowed_prefixes=(
            'neck.',
            'bbox_head.',
        ),
        type='TrainableParameterAuditHook'),
    dict(check_iters=25, type='PositiveDetectionLossHook'),
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'yolo_world',
        'newwork',
        'openworld_1.standard_ovd_components',
    ])
data_root = '/home/songziqi/project/Data/UAV/5-shots'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.67
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        filename_tmpl='iter_{}.pth',
        interval=100,
        max_keep_ckpts=-1,
        rule='greater',
        save_best='coco/bbox_mAP_50',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=None,
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_classes = ('small drone', )
eval_metainfo = dict(classes=('small drone', ))
img_scale = (
    1280,
    1280,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
last_stage_out_channels = 768
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'none'
load_from = '/home/songziqi/project/YOLO-World/newwork/openworld_1/runs/sparse_zoom_v2_m_ovd_base2_1280_e80/best_coco_bbox_mAP_epoch_80.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=20)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 80
max_iters = 1000
max_keep_ckpts = 2
metainfo = dict(classes=(
    'multirotor drone',
    'helicopter',
))
mixup_prob = 0.1
model = dict(
    backbone=dict(
        image_model=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            arch='P5',
            deepen_factor=0.67,
            frozen_stages=4,
            last_stage_out_channels=768,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            norm_eval=True,
            out_indices=(
                1,
                2,
                3,
                4,
            ),
            type='YOLOv8CSPDarknet',
            widen_factor=0.75),
        text_model=dict(
            frozen_modules=[
                'all',
            ],
            model_name=
            '/home/songziqi/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
            type='HuggingCLIPLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            cls_hidden_channels=192,
            embed_dims=512,
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            in_channels=[
                128,
                256,
                512,
                768,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=3,
            reg_hidden_channels=64,
            reg_max=16,
            type='PretrainedCompatibleYOLOWorldHeadModule',
            use_bn_head=True,
            widen_factor=0.75),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5,
            strides=[
                4,
                8,
                16,
                32,
            ],
            type='mmdet.MlvlPointGenerator'),
        type='YOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            255,
            255,
            255,
        ],
        type='mmdet.DetDataPreprocessor'),
    mm_neck=True,
    neck=dict(
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        deepen_factor=0.67,
        embed_channels=[
            128,
            256,
            384,
        ],
        guide_channels=512,
        in_channels=[
            256,
            512,
            768,
        ],
        num_csp_blocks=3,
        num_heads=[
            4,
            8,
            12,
        ],
        out_channels=[
            256,
            512,
            768,
        ],
        p2_in_channels=128,
        p2_out_channels=128,
        refinement_blocks=2,
        reparameterized=False,
        router_channels=64,
        sparse_ratio=0.25,
        text_channels=512,
        type='TextGuidedSparseYOLOWorldPAFPN',
        widen_factor=0.75,
        window_size=8),
    num_test_classes=1,
    num_train_classes=3,
    router_coverage_weight=0.5,
    router_loss_weight=1.0,
    small_object_max_size=96.0,
    sparse_budget_weight=0.1,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=3,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='FrozenBackboneTextGuidedSparseOVDDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            1280,
            1280,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -640,
            -640,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
]
neck_embed_channels = [
    128,
    256,
    384,
]
neck_num_heads = [
    4,
    8,
    12,
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 80
num_det_layers = 3
num_test_classes = 1
num_train_classes = 3
num_training_classes = 2
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0, norm_type=2),
    constructor='YOLOWv5OptimizerConstructor',
    optimizer=dict(lr=1e-05, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 5
strides = [
    4,
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=
        '/home/songziqi/project/Data/UAV/5-shots/texts/eval_class_texts.json',
        dataset=dict(
            ann_file=
            '/home/songziqi/project/Data/UAV/5-shots/annotations/instances_test_small_drone.json',
            data_prefix=dict(img='images/test/'),
            data_root='/home/songziqi/project/Data/UAV/RGB',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=('small drone', )),
            test_mode=True,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/songziqi/project/Data/UAV/5-shots/annotations/instances_test_small_drone.json',
    classwise=True,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        1280,
        1280,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            1280,
            1280,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
text_channels = 512
text_model_name = '/home/songziqi/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
text_root = '/home/songziqi/project/Data/UAV/5-shots/texts'
text_transform = [
    dict(
        max_num_samples=2,
        num_neg_samples=(
            2,
            2,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 4
train_cfg = dict(
    max_iters=1000, type='IterBasedTrainLoop', val_begin=100, val_interval=100)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        class_text_path=
        '/home/songziqi/project/Data/UAV/5-shots/texts/train_class_texts.json',
        dataset=dict(
            ann_file=
            '/home/songziqi/project/Data/UAV/5-shots/annotations/instances_train_5shot_replay.json',
            data_prefix=dict(img='mixed_train/images/'),
            data_root='/home/songziqi/project/Data/UAV/5-shots',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'small drone',
                    'multirotor drone',
                    'helicopter',
                )),
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                1280,
                1280,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    1280,
                    1280,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100.0,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                max_translate_ratio=0.05,
                scaling_ratio_range=(
                    0.9,
                    1.1,
                ),
                type='YOLOv5RandomAffine'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=3,
                num_neg_samples=(
                    2,
                    2,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_dataset = dict(
    class_text_path=
    '/home/songziqi/project/Data/UAV/5-shots/texts/train_class_texts.json',
    dataset=dict(
        ann_file=
        '/home/songziqi/project/Data/UAV/5-shots/annotations/instances_train_5shot_replay.json',
        data_prefix=dict(img='mixed_train/images/'),
        data_root='/home/songziqi/project/Data/UAV/5-shots',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'small drone',
                'multirotor drone',
                'helicopter',
            )),
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(scale=(
            1280,
            1280,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=True,
            pad_val=dict(img=114.0),
            scale=(
                1280,
                1280,
            ),
            type='LetterResize'),
        dict(
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100.0,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            max_translate_ratio=0.05,
            scaling_ratio_range=(
                0.9,
                1.1,
            ),
            type='YOLOv5RandomAffine'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=3,
            num_neg_samples=(
                2,
                2,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
train_metainfo = dict(
    classes=(
        'small drone',
        'multirotor drone',
        'helicopter',
    ))
train_num_workers = 8
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        1280,
        1280,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            1280,
            1280,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(
            0.9,
            1.1,
        ),
        type='YOLOv5RandomAffine'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=3,
        num_neg_samples=(
            2,
            2,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        1280,
        1280,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            1280,
            1280,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=2,
        num_neg_samples=(
            2,
            2,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=
        '/home/songziqi/project/Data/UAV/5-shots/texts/eval_class_texts.json',
        dataset=dict(
            ann_file=
            '/home/songziqi/project/Data/UAV/5-shots/annotations/instances_val_small_drone.json',
            data_prefix=dict(img='images/val/'),
            data_root='/home/songziqi/project/Data/UAV/RGB',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=('small drone', )),
            test_mode=True,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset = dict(
    class_text_path=
    '/home/songziqi/project/Data/openworld_base/texts/base_class_texts.json',
    dataset=dict(
        ann_file=
        '/home/songziqi/project/Data/openworld_base/annotations/instances_val_base.json',
        data_prefix=dict(img='val_base/images/'),
        data_root='/home/songziqi/project/Data/openworld_base',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=(
            'multirotor drone',
            'helicopter',
        )),
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
val_evaluator = dict(
    ann_file=
    '/home/songziqi/project/Data/UAV/5-shots/annotations/instances_val_small_drone.json',
    classwise=True,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval = 100
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.01
widen_factor = 0.75
work_dir = '/home/songziqi/project/YOLO-World/newwork/openworld_1/5-shots/zero_shot/sparse_zoom_best80_small_drone'
