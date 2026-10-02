"""Sparse-Zoom YOLO-World V2-M base2 OVD fine-tuning for 80 epochs."""

_base_ = '../../openworld_1/yolo_world_v2_m_openworld_standard_1280.py'

custom_imports = dict(
    imports=[
        'yolo_world', 'newwork', 'openworld_1.standard_ovd_components'
    ],
    allow_failed_imports=False)

data_root = '/home/songziqi/project/Data/openworld_base'
classes = ('multirotor drone', 'helicopter')
num_training_classes = 2
num_test_classes = 2
max_epochs = 80
close_mosaic_epochs = 10
text_channels = 512
strides = [4, 8, 16, 32]
deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768
text_model_name = (
    '/home/songziqi/.cache/huggingface/hub/'
    'models--openai--clip-vit-base-patch32/snapshots/'
    '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')

load_from = (
    '/home/songziqi/project/YOLO-World/newwork/openworld_1/weights/'
    'yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft_p2_ovd.pth')

model = dict(
    _delete_=True,
    type='TextGuidedSparseOVDDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_test_classes,
    router_loss_weight=1.0,
    router_coverage_weight=0.5,
    sparse_budget_weight=0.1,
    small_object_max_size=96.0,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=last_stage_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            out_indices=(1, 2, 3, 4),
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
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
            type='PretrainedCompatibleYOLOWorldHeadModule',
            num_classes=num_training_classes,
            in_channels=[128, 256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            cls_hidden_channels=192,
            reg_hidden_channels=64,
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
            num_classes=num_training_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# P2 raises memory use substantially; batch 2 is the safe 4090 default.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(optimizer=dict(batch_size_per_gpu=2))

default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear', lr_factor=0.01, max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best='coco/bbox_mAP',
        rule='greater',
        interval=5))
custom_hooks = [
    dict(type='PositiveDetectionLossHook', check_iters=25),
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
        switch_pipeline=_base_.train_pipeline_stage2)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])
